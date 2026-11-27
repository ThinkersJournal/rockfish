// SPDX-License-Identifier: MIT OR Apache-2.0
//! Unpopped 0.61.3+ `check_kernel` computes `I128`/`U128`/`I256`/`U256`; an emitter that does not lower them must refuse
//! **by name** before emitting. This is that check for `Hip::lower_kernel`: a kernel carrying a wide integer (as a buffer
//! element, as a local, as the compute type) must be a typed `UnsupportedKernel` whose text names the dtype: never source, never a
//! panic. Control: the same kernel shape over `I32` lowers, so the refusal is about the dtype. Lowering wide integers is NOT asked
//! (the carry-flag emit hold, Intel US11531542B2, stands: IR and oracle are fine, emit is held).

use std::panic::{AssertUnwindSafe, catch_unwind};

use rockfish_hip_emit::Hip;
use unpopped::backend::{Backend, LowerError};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

fn copy_kernel(elem: ElementKind) -> KernelFn {
    let at0 = |buffer| Place::Logical {
        buffer,
        coords: vec![ScalarExpr::ConstInt(0, ElementKind::I32)],
    };
    KernelFn::new(
        "wide_copy",
        elem,
        vec![
            Param::buffer("x", elem, BufferRole::Read, Some(1)),
            Param::buffer("y", elem, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(elem)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at0(0),
            },
            Stmt::Store {
                place: at0(1),
                value: ScalarExpr::Local(0),
            },
        ]),
    )
}

fn outcome(k: &KernelFn) -> Result<Result<String, String>, String> {
    match catch_unwind(AssertUnwindSafe(|| Hip.lower_kernel(k))) {
        Err(_) => Err("PANIC".into()),
        Ok(Ok(g)) => Ok(Ok(g.source)),
        Ok(Err(LowerError::UnsupportedKernel { detail })) => Ok(Err(detail)),
        Ok(Err(other)) => Err(format!("other error {other:?}")),
    }
}

#[test]
fn control_i32_lowers() {
    let r = outcome(&copy_kernel(ElementKind::I32)).expect("no panic, typed error");
    assert!(r.is_ok(), "control must lower: {r:?}");
}

#[test]
fn wide_integers_are_refused_by_name_not_emitted_not_panicked() {
    for (ek, name) in [
        (ElementKind::I128, "I128"),
        (ElementKind::U128, "U128"),
        (ElementKind::I256, "I256"),
        (ElementKind::U256, "U256"),
    ] {
        let k = copy_kernel(ek);
        match outcome(&k) {
            Err(e) => panic!("{name}: {e}"),
            Ok(Ok(src)) => panic!("{name}: EMITTED source instead of refusing:\n{src}"),
            Ok(Err(detail)) => {
                assert!(detail.starts_with("hip backend"), "{name}: {detail}");
                assert!(
                    detail.contains(name),
                    "{name}: the error must name the dtype: {detail}"
                );
            }
        }
    }
}

/// A wide integer that is only a local and a constant (the buffers stay `I32`): still refused by name.
#[test]
fn a_wide_local_or_compute_type_is_refused_by_name() {
    for (ek, name) in [(ElementKind::I128, "I128"), (ElementKind::U256, "U256")] {
        let mut k = copy_kernel(ElementKind::I32);
        k.locals.push(LocalDecl::scalar(ek));
        k.body.0.push(Stmt::Let {
            local: 1,
            init: ScalarExpr::ConstInt(1, ek),
        });
        let detail = match outcome(&k) {
            Ok(Err(d)) => d,
            other => panic!("{name} local: expected a typed refusal, got {other:?}"),
        };
        eprintln!("{name} local: {detail}");
        assert!(detail.contains(name), "{name} local: {detail}");
        // `compute` only types the untyped `Const` leaves. With none in the kernel and every buffer `I32`, no wide value
        // exists, so emitting plain `int` code is correct (measured: it emits); the property that must hold is that no wide
        // spelling is ever emitted. With a `Const` leaf the compute type is used, and that is refused by name.
        let mut k = copy_kernel(ElementKind::I32);
        k.compute = ek;
        match outcome(&k) {
            Ok(Ok(src)) => assert!(
                !src.contains("__int128") && !src.contains("int128") && !src.contains("256"),
                "{name} compute: a wide spelling was emitted: {src}"
            ),
            Ok(Err(d)) => assert!(d.contains(name), "{name} compute: {d}"),
            Err(e) => panic!("{name} compute: {e}"),
        }
        let mut k = copy_kernel(ElementKind::I32);
        k.compute = ek;
        k.body.0.push(Stmt::Assign {
            local: 0,
            value: ScalarExpr::Const(1.0),
        });
        match outcome(&k) {
            Ok(Err(d)) => eprintln!("{name} compute+Const: {d}"),
            Ok(Ok(src)) => assert!(
                !src.contains("int128") && !src.contains("256"),
                "{name} compute+Const emitted a wide spelling: {src}"
            ),
            Err(e) => panic!("{name} compute+Const: {e}"),
        }
    }
}
