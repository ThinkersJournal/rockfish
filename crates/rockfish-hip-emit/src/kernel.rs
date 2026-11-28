// SPDX-License-Identifier: MIT OR Apache-2.0
//! The kernel-layer lowerer, slice S1 (CireSnave board 506 plan; pm-kernels 2026-11-26): the
//! skeleton. A `KernelFn` is lowered to one HIP `__global__` function.
//!
//! **Spelled:** parameters (buffers and scalars), locals, `Let`, `Load`, `Assign`, `Store`,
//! `If`, `For`, `While`, `Break`, `Continue`, `Return`, and the expressions `Const`, `ConstInt`,
//! `Local`, `Exec` (thread, block, block-dim and grid-dim ids), `Add`, `Sub`, `Mul`, `Div` at
//! a float type, and the six comparisons `CmpEq`..`CmpGe` (`(a < b) ? 1 : 0`, in the operand
//! dtype, NaN as the C operators; Unpopped's `BinaryOp` doc), at `f32`, `f64`, `i32`, `i64`, `u32` and `u64`. **Tiles (R7-D15):** with a stated
//! target the four `Tile*` statements lower to three AMD matrix atoms (`tile.rs`). **Everything else is declined by
//! name** with a typed `UnsupportedKernel` (statement, expression, place, dtype, parameter kind,
//! packed local, allocation).
//!
//! **Launch ABI (a stated contract, not checked, UNVERIFIED on a device):** the kernel is
//! `extern "C" __global__ void rockfish_hip_k_<name>(...)`. Buffer `i` is `T* __restrict__ p<i>`
//! (`const T*` for a read-only buffer) already offset to the view origin by the launcher, followed
//! by one `long long p<i>_s<k>` element stride per coordinate axis `k` (the rank is the length of
//! the `Place::Logical` coordinate list, which every use of the buffer must agree on). A scalar
//! parameter is `T l<local>`. Parameters are passed in `KernelFn::params` order, each buffer
//! followed by its strides. Text only: no HIP compiler has seen it (rig row).
//!
//! **Integer arithmetic** at a signed type wraps (computed through the unsigned type of the same
//! width) so the text has no signed-overflow undefined behaviour.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::fmt::Write as _;

use unpopped::backend::{GeneratedKernel, LowerError};
use unpopped::exec::{ExecId, Extent, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, ReduceOp, RoundMode, Saturation, ScalarExpr, UnaryOp};
use unpopped::kernel::BarrierKind;
use unpopped::kernel::{
    BufferRole, CollectiveKind, KernelFn, ParamKind, Place, ShuffleMode, Stmt, VoteKind,
};
use unpopped::kernel_interface::KernelInterface;
use unpopped_vocab::ElementKind;

use crate::tile::{Role, TilePlan};

fn decline(kernel: &KernelFn, what: impl std::fmt::Display) -> LowerError {
    LowerError::UnsupportedKernel {
        detail: format!(
            "hip backend: kernel `{}`: {what}; the S1 kernel-layer lowerer spells only the statements, expressions and dtypes listed in its module doc",
            kernel.name
        ),
    }
}

/// The C spelling and the unsigned twin (for wrapping signed arithmetic) of a spelled dtype.
fn c_type(ek: ElementKind) -> Option<(&'static str, Option<&'static str>)> {
    Some(match ek {
        ElementKind::F32 => ("float", None),
        ElementKind::F64 => ("double", None),
        ElementKind::I32 => ("int", Some("unsigned int")),
        ElementKind::I64 => ("long long", Some("unsigned long long")),
        ElementKind::U32 => ("unsigned int", None),
        ElementKind::U64 => ("unsigned long long", None),
        _ => return None,
    })
}

/// The C operator of a comparison `BinaryOp`.
const fn cmp_op(op: BinaryOp) -> Option<&'static str> {
    match op {
        BinaryOp::CmpEq => Some("=="),
        BinaryOp::CmpNe => Some("!="),
        BinaryOp::CmpLt => Some("<"),
        BinaryOp::CmpLe => Some("<="),
        BinaryOp::CmpGt => Some(">"),
        BinaryOp::CmpGe => Some(">="),
        _ => None,
    }
}

const fn is_float(ek: ElementKind) -> bool {
    matches!(ek, ElementKind::F32 | ElementKind::F64)
}

/// The leading identifier of a `Debug` rendering: the variant name.
fn variant<T: std::fmt::Debug>(v: &T) -> String {
    format!("{v:?}")
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect()
}

struct Cx<'a> {
    k: &'a KernelFn,
    /// Per buffer parameter: its rank, fixed by its first use.
    ranks: BTreeMap<u32, usize>,
    tmp: u32,
    out: String,
    /// The wavefront size of the stated target, when the caller stated one.
    wave: Option<u32>,
    /// The tile plan (R7-D15), when the kernel has `Tile` allocations and states a target that lowers them.
    tiles: Option<TilePlan>,
    /// `f16` is spelled (`_Float16`) only while a tile statement, a tile-kernel parameter or a tile-kernel `Shared` array is
    /// being written: it is a tile operand type, not a general local type.
    f16_ok: Cell<bool>,
}

impl Cx<'_> {
    fn ty_of(&self, e: &ScalarExpr) -> Result<ElementKind, LowerError> {
        match e {
            ScalarExpr::Const(_) => Ok(self.k.compute),
            ScalarExpr::ConstInt(_, ek) => Ok(*ek),
            ScalarExpr::Exec(_) => Ok(ElementKind::U32),
            ScalarExpr::Local(l) => {
                let d = self
                    .k
                    .locals
                    .get(*l as usize)
                    .ok_or_else(|| decline(self.k, format!("local {l} is not declared")))?;
                if d.lanes != 1 {
                    return Err(decline(
                        self.k,
                        format!("packed local {l} ({} lanes)", d.lanes),
                    ));
                }
                Ok(d.elem)
            }
            ScalarExpr::Binary(op, a, _) if cmp_op(*op).is_some() => self.ty_of(a),
            ScalarExpr::Binary(BinaryOp::Max | BinaryOp::Min, a, _)
            | ScalarExpr::Unary(UnaryOp::Neg | UnaryOp::Abs, a)
            | ScalarExpr::Select(_, a, _) => self.ty_of(a),
            ScalarExpr::Cast(_, spec) => Ok(spec.to),
            ScalarExpr::Add(a, _)
            | ScalarExpr::Sub(a, _)
            | ScalarExpr::Mul(a, _)
            | ScalarExpr::Div(a, _) => self.ty_of(a),
            ScalarExpr::Unary(op @ (UnaryOp::Popc | UnaryOp::Clz | UnaryOp::Brev), _) => {
                Err(decline(
                    self.k,
                    format!(
                        "unary op `{op:?}` (no lowering decided; not lowered to `__popc`/`__clz`/`__brev`)"
                    ),
                ))
            }
            other => Err(decline(self.k, format!("expression `{}`", variant(other)))),
        }
    }

    fn spelled(
        &self,
        ek: ElementKind,
        what: &str,
    ) -> Result<(&'static str, Option<&'static str>), LowerError> {
        if ek == ElementKind::F16 && self.f16_ok.get() {
            return Ok(("_Float16", None));
        }
        c_type(ek).ok_or_else(|| decline(self.k, format!("dtype {ek:?} ({what}) is not spelled")))
    }

    fn expr(&self, e: &ScalarExpr) -> Result<String, LowerError> {
        match e {
            ScalarExpr::Const(v) => {
                let ek = self.k.compute;
                self.spelled(ek, "the kernel's compute dtype")?;
                if !is_float(ek) {
                    return Err(decline(
                        self.k,
                        format!("`Const` at the integer compute dtype {ek:?}"),
                    ));
                }
                if !v.is_finite() {
                    return Err(decline(self.k, "a non-finite `Const`"));
                }
                Ok(if ek == ElementKind::F32 {
                    format!("{:?}f", *v as f32)
                } else {
                    format!("{v:?}")
                })
            }
            ScalarExpr::ConstInt(v, ek) => {
                let (t, _) = self.spelled(*ek, "a `ConstInt`")?;
                if is_float(*ek) {
                    return Err(decline(
                        self.k,
                        format!("`ConstInt` at the float dtype {ek:?}"),
                    ));
                }
                Ok(format!("(({t}){v}LL)"))
            }
            ScalarExpr::Local(l) => {
                self.ty_of(e)?;
                Ok(format!("l{l}"))
            }
            ScalarExpr::Exec(id) => {
                let s = match id {
                    ExecId::ThreadIdx(a) if *a < 3 => {
                        format!("threadIdx.{}", ['x', 'y', 'z'][*a as usize])
                    }
                    ExecId::BlockIdx(a) if *a < 3 => {
                        format!("blockIdx.{}", ['x', 'y', 'z'][*a as usize])
                    }
                    ExecId::BlockDim(a) if *a < 3 => {
                        format!("blockDim.{}", ['x', 'y', 'z'][*a as usize])
                    }
                    ExecId::GridDim(a) if *a < 3 => {
                        format!("gridDim.{}", ['x', 'y', 'z'][*a as usize])
                    }
                    other => {
                        return Err(decline(
                            self.k,
                            format!("execution id `{}`", variant(other)),
                        ));
                    }
                };
                Ok(format!("((unsigned int){s})"))
            }
            ScalarExpr::Binary(op, a, b) if cmp_op(*op).is_some() => {
                let ek = self.ty_of(e)?;
                let (t, _) = self.spelled(ek, "a comparison operand")?;
                let c = cmp_op(*op).unwrap_or("==");
                Ok(format!(
                    "(({t})(({}) {c} ({}) ? 1 : 0))",
                    self.expr(a)?,
                    self.expr(b)?
                ))
            }
            ScalarExpr::Binary(op @ (BinaryOp::Max | BinaryOp::Min), a, b) => {
                let ek = self.ty_of(e)?;
                self.spelled(ek, "a `Max`/`Min`")?;
                let (x, y) = (self.expr(a)?, self.expr(b)?);
                let (cmp, name) = if *op == BinaryOp::Max {
                    (">=", "Max")
                } else {
                    ("<=", "Min")
                };
                let _ = name;
                // Operands appear several times: every expression this lowerer builds is pure.
                Ok(if is_float(ek) {
                    // NaN-propagating, the first operand on ties (the IR's `Max`/`Min`, not C `fmax`).
                    format!("({x} != {x} ? {x} : ({y} != {y} ? {y} : ({x} {cmp} {y} ? {x} : {y})))")
                } else {
                    let c = if *op == BinaryOp::Max { ">" } else { "<" };
                    format!("(({x}) {c} ({y}) ? ({x}) : ({y}))")
                })
            }
            ScalarExpr::Unary(UnaryOp::Neg, a) => {
                let ek = self.ty_of(e)?;
                let (t, unsigned) = self.spelled(ek, "a `Neg`")?;
                let x = self.expr(a)?;
                Ok(match (is_float(ek), unsigned) {
                    (true, _) => format!("(-{x})"),
                    (false, Some(u)) => format!("(({t})(0u - ({u})({x})))"),
                    (false, None) => format!("(({t})(({t})0 - ({x})))"),
                })
            }
            ScalarExpr::Unary(UnaryOp::Abs, a) => {
                let ek = self.ty_of(e)?;
                if !is_float(ek) {
                    return Err(decline(
                        self.k,
                        format!("integer `Abs` at {ek:?} (the MIN case is not decided in S1b)"),
                    ));
                }
                let f = if ek == ElementKind::F32 {
                    "fabsf"
                } else {
                    "fabs"
                };
                Ok(format!("{f}({})", self.expr(a)?))
            }
            ScalarExpr::Cast(a, spec) => self.cast(a, spec.to, spec.round, spec.sat),
            ScalarExpr::Select(c, a, b) => {
                let ek = self.ty_of(e)?;
                if !is_float(ek) {
                    return Err(decline(
                        self.k,
                        format!("`Select` at {ek:?} (float only, as the IR admits it)"),
                    ));
                }
                self.spelled(ek, "a `Select`")?;
                Ok(format!(
                    "(({}) != 0 ? ({}) : ({}))",
                    self.expr(c)?,
                    self.expr(a)?,
                    self.expr(b)?
                ))
            }
            ScalarExpr::Add(a, b) => self.arith(e, a, b, '+'),
            ScalarExpr::Sub(a, b) => self.arith(e, a, b, '-'),
            ScalarExpr::Mul(a, b) => self.arith(e, a, b, '*'),
            ScalarExpr::Div(a, b) => {
                let ek = self.ty_of(e)?;
                if !is_float(ek) {
                    return Err(decline(
                        self.k,
                        format!("integer `Div` at {ek:?} (total-division rule not lowered in S1)"),
                    ));
                }
                self.spelled(ek, "a `Div`")?;
                Ok(format!("({} / {})", self.expr(a)?, self.expr(b)?))
            }
            ScalarExpr::Unary(op @ (UnaryOp::Popc | UnaryOp::Clz | UnaryOp::Brev), _) => {
                Err(decline(
                    self.k,
                    format!(
                        "unary op `{op:?}` (no lowering decided; not lowered to `__popc`/`__clz`/`__brev`)"
                    ),
                ))
            }
            other => Err(decline(self.k, format!("expression `{}`", variant(other)))),
        }
    }

    /// `Cast` between an integer and a float type (S1c). Int to float is C's conversion (round to
    /// nearest even, the default); float to int is `TowardZero` with `Clamp`, NaN to 0, spelled
    /// with explicit bounds so no input reaches C's undefined behaviour. Every other pair, rounding
    /// mode and saturation is refused by name.
    fn cast(
        &self,
        a: &ScalarExpr,
        to: ElementKind,
        round: RoundMode,
        sat: Saturation,
    ) -> Result<String, LowerError> {
        let from = self.ty_of(a)?;
        self.spelled(from, "a cast source")?;
        let (tt, _) = self.spelled(to, "a cast destination")?;
        let x = self.expr(a)?;
        match (is_float(from), is_float(to)) {
            (false, true) => {
                if round != RoundMode::NearestEven {
                    return Err(decline(
                        self.k,
                        format!(
                            "an int to float `Cast` rounding {round:?} (only NearestEven is spelled)"
                        ),
                    ));
                }
                Ok(format!("(({tt})({x}))"))
            }
            (true, false) => {
                if round != RoundMode::TowardZero || sat != Saturation::Clamp {
                    return Err(decline(
                        self.k,
                        format!(
                            "a float to int `Cast` with {round:?} and {sat:?} (only TowardZero with Clamp is spelled)"
                        ),
                    ));
                }
                let (bits, signed) = match to {
                    ElementKind::I32 => (32, true),
                    ElementKind::I64 => (64, true),
                    ElementKind::U32 => (32, false),
                    _ => (64, false),
                };
                let two_pow = |n: u32| format!("{:?}", 2f64.powi(n as i32));
                let sfx = if from == ElementKind::F32 { "f" } else { "" };
                let (hi, lo, max, min) = if signed {
                    let w = bits - 1;
                    let (max, min) = if bits == 32 {
                        ("2147483647", "(-2147483647 - 1)")
                    } else {
                        ("9223372036854775807LL", "(-9223372036854775807LL - 1)")
                    };
                    (
                        two_pow(w),
                        format!("-{}", two_pow(w)),
                        max.to_string(),
                        min.to_string(),
                    )
                } else {
                    let max = if bits == 32 {
                        "4294967295u"
                    } else {
                        "18446744073709551615ull"
                    };
                    (
                        two_pow(bits),
                        "0.0".to_string(),
                        max.to_string(),
                        "0".to_string(),
                    )
                };
                Ok(format!(
                    "(({x}) != ({x}) ? ({tt})0 : (({x}) >= {hi}{sfx} ? ({tt}){max} : (({x}) <= {lo}{sfx} ? ({tt}){min} : ({tt})({x}))))"
                ))
            }
            _ => Err(decline(
                self.k,
                format!(
                    "a `Cast` from {from:?} to {to:?} (only int to float and float to int are spelled)"
                ),
            )),
        }
    }

    fn arith(
        &self,
        e: &ScalarExpr,
        a: &ScalarExpr,
        b: &ScalarExpr,
        op: char,
    ) -> Result<String, LowerError> {
        let ek = self.ty_of(e)?;
        let (t, unsigned) = self.spelled(ek, "an arithmetic result")?;
        let (a, b) = (self.expr(a)?, self.expr(b)?);
        Ok(match unsigned {
            // Signed: wrap through the unsigned twin.
            Some(u) => format!("(({t})(({u})({a}) {op} ({u})({b})))"),
            None => format!("({a} {op} {b})"),
        })
    }

    /// The element address of a `Place::Logical`, and the buffer's element type.
    fn place(&mut self, p: &Place) -> Result<(String, ElementKind), LowerError> {
        match p {
            Place::Logical { buffer, coords } => {
                let param = self.k.params.get(*buffer as usize).ok_or_else(|| {
                    decline(self.k, format!("buffer parameter {buffer} does not exist"))
                })?;
                let ParamKind::Buffer { elem, .. } = &param.kind else {
                    return Err(decline(
                        self.k,
                        format!("parameter {buffer} is not a buffer"),
                    ));
                };
                self.spelled(*elem, "a buffer element")?;
                if coords.is_empty() {
                    return Err(decline(self.k, "a rank-0 `Place::Logical`"));
                }
                let rank = *self.ranks.entry(*buffer).or_insert(coords.len());
                if rank != coords.len() {
                    return Err(decline(
                        self.k,
                        format!(
                            "buffer {buffer} is addressed with {rank} and {} coordinates",
                            coords.len()
                        ),
                    ));
                }
                let mut terms = Vec::new();
                for (axis, c) in coords.iter().enumerate() {
                    let ek = self.ty_of(c)?;
                    if is_float(ek) {
                        return Err(decline(self.k, "a float coordinate"));
                    }
                    terms.push(format!(
                        "(long long)({}) * p{buffer}_s{axis}",
                        self.expr(c)?
                    ));
                }
                Ok((format!("p{buffer}[{}]", terms.join(" + ")), *elem))
            }
            Place::Alloc { alloc, index } => {
                let a =
                    self.k.allocs.get(*alloc as usize).ok_or_else(|| {
                        decline(self.k, format!("allocation {alloc} does not exist"))
                    })?;
                let ek = self.ty_of(index)?;
                if is_float(ek) {
                    return Err(decline(self.k, "a float allocation index"));
                }
                Ok((
                    format!("s{alloc}[(long long)({})]", self.expr(index)?),
                    a.elem,
                ))
            }
            other => Err(decline(self.k, format!("place `{}`", variant(other)))),
        }
    }

    /// The ruled subgroup collectives (board 506 "A", 2026-11-26): `Shuffle`, `Vote`, `Reduce`
    /// and `Broadcast` at `Scope::Subgroup` on a stated RDNA or CDNA target. Every source lane is
    /// computed here, in range, and handed to HIP's `__shfl` with an in-range `srcLane`, so the
    /// IR's edge rules (docs/design-kernel-collectives-edges.md) are spelled, never inherited
    /// from a device. UNMEASURED on a device; text parsed by clang only.
    fn collective(
        &mut self,
        kind: &CollectiveKind,
        value: &ScalarExpr,
        result: u32,
        scope: Scope,
        d: usize,
    ) -> Result<(), LowerError> {
        let k = self.k;
        let marker = "declined by name: board 434 C, US8539204B2 (only subgroup-scope shuffles, votes, reductions and broadcasts are lifted, board 506 A)";
        if scope != Scope::Subgroup {
            return Err(decline(
                k,
                format!("a `Collective` at {scope:?} scope is {marker}"),
            ));
        }
        let Some(n) = self.wave else {
            return Err(decline(
                k,
                format!(
                    "a `Collective` needs a target: call `Hip::lower_kernel_for` (the plain `lower_kernel` states none; {marker})"
                ),
            ));
        };
        let sub = k.launch.and_then(|l| l.subgroup);
        let block_ok = k.launch.is_some_and(|l| {
            l.block.iter().map(|&b| u64::from(b)).product::<u64>() % u64::from(n) == 0
        });
        if !sub.is_some_and(|s| s.min == n && s.max == n) || !block_ok {
            return Err(decline(
                k,
                format!(
                    "the launch does not state subgroup size {n} exactly (and a block of a multiple of {n} threads), which this target's wavefront needs"
                ),
            ));
        }
        let vt = self.ty_of(value)?;
        let (t, unsigned) = self.spelled(vt, "a collective value")?;
        let v = self.expr(value)?;
        let rt = self.ty_of(&ScalarExpr::Local(result))?;
        let id = self.tmp;
        self.tmp += 1;
        let nn = i64::from(n);
        let lane = format!("lane{id}");
        let int_operand = |this: &Self, e: &ScalarExpr| -> Result<String, LowerError> {
            let ek = this.ty_of(e)?;
            if !matches!(ek, ElementKind::I32 | ElementKind::I64 | ElementKind::U32) {
                return Err(decline(
                    k,
                    format!("a lane operand of type {ek:?} (I32, I64 and U32 are spelled)"),
                ));
            }
            Ok(format!("(long long)({})", this.expr(e)?))
        };
        let mut pre = vec![
            format!("const {t} v{id} = {v};"),
            format!(
                "const long long {lane} = (long long)((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) % {n}u);"
            ),
        ];
        let rhs = match kind {
            CollectiveKind::Shuffle { mode, operand } => {
                if rt != vt {
                    return Err(decline(
                        k,
                        "a shuffle result of another type than its value",
                    ));
                }
                let o = int_operand(self, operand)?;
                let src = match mode {
                    ShuffleMode::Idx => format!("((({o}) % {nn}LL) + {nn}LL) % {nn}LL"),
                    ShuffleMode::Up => format!("{lane} - ({o})"),
                    ShuffleMode::Down => format!("{lane} + ({o})"),
                    ShuffleMode::Xor => format!("{lane} ^ ({o})"),
                    other => return Err(decline(k, format!("shuffle mode `{}`", variant(other)))),
                };
                pre.push(format!("const long long src{id} = {src};"));
                // Out of range: the lane's own value (the IR rule, spelled here).
                format!(
                    "__shfl(v{id}, (int)((src{id} >= 0 && src{id} < {nn}LL) ? src{id} : {lane}), {n})"
                )
            }
            CollectiveKind::Broadcast { lane: l } => {
                if rt != vt {
                    return Err(decline(
                        k,
                        "a broadcast result of another type than its value",
                    ));
                }
                let ScalarExpr::ConstInt(c, _) = l else {
                    return Err(decline(
                        k,
                        "a broadcast from a lane that is not a constant (the IR states no edge rule for it)",
                    ));
                };
                if !(0..nn).contains(c) {
                    return Err(decline(
                        k,
                        format!("a broadcast from lane {c}, outside 0..{n}"),
                    ));
                }
                format!("__shfl(v{id}, {c}, {n})")
            }
            CollectiveKind::Vote(vote) => {
                let (rtn, _) = self.spelled(rt, "a vote result")?;
                if is_float(rt) {
                    return Err(decline(k, "a float vote result"));
                }
                match vote {
                    VoteKind::All => format!("(({rtn})__all(v{id} != 0))"),
                    VoteKind::Any => format!("(({rtn})__any(v{id} != 0))"),
                    VoteKind::Ballot => {
                        if n != 32 {
                            return Err(decline(
                                k,
                                format!(
                                    "a `Ballot` on a wavefront of {n} lanes: the IR defines it for 32 lanes and HIP's `__ballot` returns 64 bits, so it is lowered at wave size 32 only"
                                ),
                            ));
                        }
                        format!("(({rtn})(unsigned int)__ballot(v{id} != 0))")
                    }
                    other => return Err(decline(k, format!("vote `{}`", variant(other)))),
                }
            }
            CollectiveKind::Reduce(op) => {
                if rt != vt {
                    return Err(decline(
                        k,
                        "a reduction result of another type than its value",
                    ));
                }
                if is_float(vt) {
                    return Err(decline(
                        k,
                        "a float reduction (the butterfly order is not the IR's lane order, and the float tolerance is not decided for it)",
                    ));
                }
                let step = match op {
                    ReduceOp::Sum | ReduceOp::Prod => {
                        let o = if matches!(op, ReduceOp::Sum) {
                            '+'
                        } else {
                            '*'
                        };
                        match unsigned {
                            Some(u) => format!("({t})(({u})r{id} {o} ({u})t{id})"),
                            None => format!("r{id} {o} t{id}"),
                        }
                    }
                    ReduceOp::Max => format!("(r{id} > t{id} ? r{id} : t{id})"),
                    ReduceOp::Min => format!("(r{id} < t{id} ? r{id} : t{id})"),
                    ReduceOp::And => format!("r{id} & t{id}"),
                    ReduceOp::Or => format!("r{id} | t{id}"),
                    ReduceOp::Xor => format!("r{id} ^ t{id}"),
                    other => return Err(decline(k, format!("reduce op `{}`", variant(other)))),
                };
                let (rtn0, _) = self.spelled(rt, "a reduction result")?;
                self.line(d, &format!("{rtn0} l{result};"));
                self.line(d, "{");
                for l in &pre[..1] {
                    self.line(d + 1, l);
                }
                self.line(d + 1, &format!("{t} r{id} = v{id};"));
                self.line(
                    d + 1,
                    &format!("for (int o{id} = {n} / 2; o{id} > 0; o{id} >>= 1) {{"),
                );
                self.line(
                    d + 2,
                    &format!("const {t} t{id} = __shfl_xor(r{id}, o{id}, {n});"),
                );
                self.line(d + 2, &format!("r{id} = {step};"));
                self.line(d + 1, "}");
                let (rtn, _) = self.spelled(rt, "a reduction result")?;
                self.line(d + 1, &format!("l{result} = ({rtn})r{id};"));
                self.line(d, "}");
                return Ok(());
            }
            other => {
                return Err(decline(
                    k,
                    format!("a `Collective` of kind `{}` is {marker}", variant(other)),
                ));
            }
        };
        let (rtn, _) = self.spelled(rt, "a collective result")?;
        // The result local is declared at the statement's own scope; the temporaries sit in a
        // block of their own so repeated collectives do not collide.
        self.line(d, &format!("{rtn} l{result};"));
        self.line(d, "{");
        for l in &pre {
            self.line(d + 1, l);
        }
        self.line(d + 1, &format!("l{result} = {rhs};"));
        self.line(d, "}");
        Ok(())
    }

    fn block(&mut self, b: &unpopped::kernel::Block, depth: usize) -> Result<(), LowerError> {
        for s in &b.0 {
            self.stmt(s, depth)?;
        }
        Ok(())
    }

    fn line(&mut self, depth: usize, s: &str) {
        let _ = writeln!(self.out, "{}{s}", "    ".repeat(depth));
    }

    fn stmt(&mut self, s: &Stmt, d: usize) -> Result<(), LowerError> {
        match s {
            Stmt::Let { local, init } => {
                let ek = self.ty_of(&ScalarExpr::Local(*local))?;
                let (t, _) = self.spelled(ek, "a local")?;
                let v = self.expr(init)?;
                self.line(d, &format!("{t} l{local} = {v};"));
            }
            Stmt::Load { local, place } => {
                let ek = self.ty_of(&ScalarExpr::Local(*local))?;
                let (t, _) = self.spelled(ek, "a local")?;
                let (addr, belem) = self.place(place)?;
                if belem != ek {
                    return Err(decline(
                        self.k,
                        format!("`Load` of a {belem:?} element into a {ek:?} local"),
                    ));
                }
                self.line(d, &format!("{t} l{local} = {addr};"));
            }
            Stmt::MaskedLoad {
                local,
                place,
                mask,
                fill,
            } => {
                let ek = self.ty_of(&ScalarExpr::Local(*local))?;
                let (t, _) = self.spelled(ek, "a local")?;
                let (addr, belem) = self.place(place)?;
                if belem != ek {
                    return Err(decline(
                        self.k,
                        format!("`MaskedLoad` of a {belem:?} element into a {ek:?} local"),
                    ));
                }
                let (m, f) = (self.expr(mask)?, self.expr(fill)?);
                // The untaken arm of a C ternary is not evaluated: a false mask touches no memory.
                self.line(d, &format!("{t} l{local} = (({m}) != 0) ? {addr} : ({f});"));
            }
            Stmt::MaskedStore { place, value, mask } => {
                let (addr, _) = self.place(place)?;
                if let Place::Logical { buffer, .. } = place
                    && matches!(&self.k.params[*buffer as usize].kind, ParamKind::Buffer { role, .. } if *role == BufferRole::Read)
                {
                    return Err(decline(self.k, "a `MaskedStore` to a read-only buffer"));
                }
                let (m, v) = (self.expr(mask)?, self.expr(value)?);
                self.line(d, &format!("if (({m}) != 0) {{ {addr} = {v}; }}"));
            }
            Stmt::Assign { local, value } => {
                self.ty_of(&ScalarExpr::Local(*local))?;
                let v = self.expr(value)?;
                self.line(d, &format!("l{local} = {v};"));
            }
            Stmt::Store { place, value } => {
                let (addr, _) = self.place(place)?;
                let role_ok = match place {
                    Place::Logical { buffer, .. } => matches!(
                        &self.k.params[*buffer as usize].kind,
                        ParamKind::Buffer { role, .. } if *role != BufferRole::Read
                    ),
                    _ => true,
                };
                if !role_ok {
                    return Err(decline(self.k, "a `Store` to a read-only buffer"));
                }
                let v = self.expr(value)?;
                self.line(d, &format!("{addr} = {v};"));
            }
            Stmt::If { cond, then, els } => {
                let c = self.expr(cond)?;
                self.line(d, &format!("if (({c}) != 0) {{"));
                self.block(then, d + 1)?;
                if els.0.is_empty() {
                    self.line(d, "}");
                } else {
                    self.line(d, "} else {");
                    self.block(els, d + 1)?;
                    self.line(d, "}");
                }
            }
            Stmt::For {
                local,
                start,
                end,
                step,
                body,
                ..
            } => {
                let ek = self.ty_of(&ScalarExpr::Local(*local))?;
                let (t, _) = self.spelled(ek, "a loop counter")?;
                if is_float(ek) {
                    return Err(decline(self.k, "a float loop counter"));
                }
                let n = self.tmp;
                self.tmp += 1;
                let (a, b, c) = (self.expr(start)?, self.expr(end)?, self.expr(step)?);
                self.line(d, "{");
                self.line(
                    d + 1,
                    &format!("const {t} lo{n} = {a}, hi{n} = {b}, st{n} = {c};"),
                );
                // A zero step traps in the oracle; here it would not terminate, so it ends the loop.
                self.line(
                    d + 1,
                    &format!("for ({t} l{local} = lo{n}; st{n} != 0 && (st{n} > 0 ? l{local} < hi{n} : l{local} > hi{n}); l{local} += st{n}) {{"),
                );
                self.block(body, d + 2)?;
                self.line(d + 1, "}");
                self.line(d, "}");
            }
            Stmt::While { cond, body } => {
                let c = self.expr(cond)?;
                self.line(d, &format!("while (({c}) != 0) {{"));
                self.block(body, d + 1)?;
                self.line(d, "}");
            }
            Stmt::Barrier { scope, kind } => match (scope, kind) {
                // A plain block barrier: HIP's `__syncthreads()`. Named, split and every other
                // scope are not lowered (named, cluster and device barriers are held).
                (Scope::Block, BarrierKind::Full) => self.line(d, "__syncthreads();"),
                (sc, kd) => {
                    return Err(decline(
                        self.k,
                        format!(
                            "a `Barrier` of scope {sc:?} and kind `{}` (only Block/Full is lowered)",
                            variant(kd)
                        ),
                    ));
                }
            },
            // Unpopped 0.64.0 `Assume`: a value fact the emitter "may lower to an optimiser hint or ignore".
            // We ignore it (emit nothing); a `__builtin_assume` lowering is a later row. Decision, not an ISA fact.
            Stmt::Assume { .. } => {}
            Stmt::Break => self.line(d, "break;"),
            Stmt::Continue => self.line(d, "continue;"),
            Stmt::Return => self.line(d, "return;"),
            Stmt::Collective {
                kind,
                value,
                result,
                scope,
            } => {
                self.collective(kind, value, *result, *scope, d)?;
            }
            Stmt::TileLoad { .. }
            | Stmt::TileFill { .. }
            | Stmt::TileMma { .. }
            | Stmt::TileStore { .. }
                if self.tiles.is_some() =>
            {
                self.f16_ok.set(true);
                let r = self.tile_stmt(s, d);
                self.f16_ok.set(false);
                r?;
            }
            other => return Err(decline(self.k, format!("statement `{}`", variant(other)))),
        }
        Ok(())
    }
}

/// Lower one kernel-layer function (S1).
pub(crate) fn lower(
    kernel: &KernelFn,
    wave: Option<u32>,
    arch: Option<&str>,
) -> Result<GeneratedKernel, LowerError> {
    // The checker is the only gate on data (`Backend::lower_kernel` doc): a kernel it refuses is
    // not lowered, whatever it contains.
    unpopped::kernel::check_kernel(kernel).map_err(|e| LowerError::UnsupportedKernel {
        detail: format!(
            "hip backend: kernel `{}`: the Unpopped checker refuses it ({e:?})",
            kernel.name
        ),
    })?;
    // Tiles (R7-D15): planned first so a refusal names the tile, not the allocation. Needs a stated target.
    let tiles = match (wave, arch) {
        (Some(w), Some(a)) if kernel.allocs.iter().any(|x| x.space == MemorySpace::Tile) => {
            let plan = crate::tile::plan(kernel, w, a).map_err(|e| decline(kernel, e))?;
            let sub = kernel.launch.and_then(|l| l.subgroup);
            let block_ok = kernel.launch.is_some_and(|l| {
                l.block.iter().map(|&b| u64::from(b)).product::<u64>() % u64::from(w) == 0
            });
            if !sub.is_some_and(|s| s.min == w && s.max == w) || !block_ok {
                return Err(decline(
                    kernel,
                    format!(
                        "the launch does not state subgroup size {w} exactly (and a block of a multiple of {w} threads), which a tile on `{a}` needs"
                    ),
                ));
            }
            Some(plan)
        }
        _ => None,
    };
    // Static `Shared` arrays only: `__shared__ T s<i>[n];`. Dynamic extents, private, constant,
    // tensor and global allocations are refused by name.
    let mut shared = String::new();
    for (i, a) in kernel.allocs.iter().enumerate() {
        if a.space == MemorySpace::Tile && tiles.is_some() {
            continue; // lowered as per-lane fragments, declared with the body
        }
        let (MemorySpace::Shared, Extent::Static(n)) = (a.space, a.extent) else {
            return Err(decline(
                kernel,
                format!(
                    "allocation `{}` in {:?} with extent {:?} (only static Shared is lowered)",
                    a.name, a.space, a.extent
                ),
            ));
        };
        let c = if a.elem == ElementKind::F16 && tiles.is_some() {
            Some(("_Float16", None))
        } else {
            c_type(a.elem)
        };
        let (t, _): (&str, Option<&str>) = c.ok_or_else(|| {
            decline(
                kernel,
                format!(
                    "dtype {:?} (allocation `{}`) is not spelled",
                    a.elem, a.name
                ),
            )
        })?;
        let _ = writeln!(shared, "    __shared__ {t} s{i}[{n}];");
    }
    if kernel.result.is_some() || !kernel.results.is_empty() {
        return Err(decline(
            kernel,
            "a function with results (a callee) is not a launchable entry",
        ));
    }
    let mut cx = Cx {
        k: kernel,
        ranks: BTreeMap::new(),
        tmp: 0,
        out: String::new(),
        wave,
        tiles,
        f16_ok: Cell::new(false),
    };
    cx.tile_prologue();
    cx.block(&kernel.body, 1)?;
    let body = std::mem::take(&mut cx.out);

    cx.f16_ok.set(cx.tiles.is_some());
    let mut sig = Vec::new();
    for (i, p) in kernel.params.iter().enumerate() {
        match &p.kind {
            ParamKind::Buffer { elem, role, .. } => {
                let (t, _) = cx.spelled(*elem, "a buffer parameter")?;
                let c = if *role == BufferRole::Read {
                    "const "
                } else {
                    ""
                };
                sig.push(format!("{c}{t}* __restrict__ p{i}"));
                for axis in 0..cx.ranks.get(&(i as u32)).copied().unwrap_or(0) {
                    sig.push(format!("long long p{i}_s{axis}"));
                }
            }
            ParamKind::Scalar { ty, local } => {
                let (t, _) = cx.spelled(*ty, "a scalar parameter")?;
                if is_float(*ty) != is_float(cx.ty_of(&ScalarExpr::Local(*local))?) {
                    return Err(decline(
                        kernel,
                        "a scalar parameter whose local has another type",
                    ));
                }
                sig.push(format!("{t} l{local}"));
            }
            other => {
                return Err(decline(
                    kernel,
                    format!("parameter kind `{}`", variant(other)),
                ));
            }
        }
    }

    let name = format!("rockfish_hip_k_{}", kernel.name);
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(decline(kernel, "a kernel name that is not a C identifier"));
    }
    let mut s = String::from("#include <hip/hip_runtime.h>\n\n");
    s.push_str(&cx.tile_typedefs());
    let _ = writeln!(s, "extern \"C\" __global__ void {name}(");
    let _ = writeln!(s, "    {}", sig.join(",\n    "));
    s.push_str(") {\n");
    s.push_str(&shared);
    s.push_str(&body);
    s.push_str("}\n");
    Ok(GeneratedKernel::new(name.clone(), s)
        .with_interface(KernelInterface::of(kernel).with_entry_symbol(name)))
}

/// The tile statements (R7-D15; the lane maps and the atom table are in `tile.rs`).
impl Cx<'_> {
    /// The fragment typedefs the planned atoms need, each once, ahead of the kernel.
    fn tile_typedefs(&self) -> String {
        let mut seen = Vec::new();
        if let Some(p) = &self.tiles {
            for (atom, _) in p.tiles.values() {
                for t in atom.typedefs() {
                    if !seen.contains(t) {
                        seen.push(*t);
                    }
                }
            }
        }
        seen.iter().map(|t| format!("{t}\n")).collect::<String>()
            + if seen.is_empty() { "" } else { "\n" }
    }

    /// The lane id and one fragment variable per tile, at the top of the body. A tile is held by the whole subgroup; this lane
    /// holds the elements the atom's lane map gives it.
    fn tile_prologue(&mut self) {
        let Some(p) = self.tiles.take() else { return };
        self.line(1, "const int rf_lane = (int)__lane_id();");
        for (t, (atom, role)) in &p.tiles {
            let (ty, _) = atom.fragment(*role);
            self.line(1, &format!("{ty} t{t} = {{}};"));
        }
        self.tiles = Some(p);
    }

    /// `t<tile>` for a scalar fragment, `t<tile>[e]` for a vector one.
    fn frag(tile: u32, atom: crate::tile::Atom, role: Role, e: usize) -> String {
        if atom.fragment(role).1 == 1 {
            format!("t{tile}")
        } else {
            format!("t{tile}[{e}]")
        }
    }

    /// The element of a tile place at `off` elements from the element the place names: a one-index buffer place or a `Shared`
    /// allocation place (the checker allows no other).
    fn tile_elem(&mut self, p: &Place, off: &str) -> Result<String, LowerError> {
        match p {
            Place::Logical { buffer, coords } if coords.len() == 1 => {
                let param = self.k.params.get(*buffer as usize).ok_or_else(|| {
                    decline(self.k, format!("buffer parameter {buffer} does not exist"))
                })?;
                let ParamKind::Buffer { elem, .. } = &param.kind else {
                    return Err(decline(
                        self.k,
                        format!("parameter {buffer} is not a buffer"),
                    ));
                };
                self.spelled(*elem, "a tile buffer element")?;
                let rank = *self.ranks.entry(*buffer).or_insert(1);
                if rank != 1 {
                    return Err(decline(
                        self.k,
                        format!("buffer {buffer} is addressed with {rank} and 1 coordinates"),
                    ));
                }
                let c = self.expr(&coords[0])?;
                Ok(format!(
                    "p{buffer}[((long long)({c}) + {off}) * p{buffer}_s0]"
                ))
            }
            Place::Alloc { alloc, index } => {
                let c = self.expr(index)?;
                Ok(format!("s{alloc}[(long long)({c}) + {off}]"))
            }
            other => Err(decline(
                self.k,
                format!("a tile place `{}`", variant(other)),
            )),
        }
    }

    fn tile_stmt(&mut self, s: &Stmt, d: usize) -> Result<(), LowerError> {
        let plan_of = |cx: &Self, t: u32| {
            cx.tiles
                .as_ref()
                .and_then(|p| p.tiles.get(&t).copied())
                .ok_or_else(|| decline(cx.k, format!("tile {t} has no planned layout")))
        };
        match s {
            Stmt::TileLoad {
                d: t,
                src,
                elements_per_row,
                origin,
            }
            | Stmt::TileStore {
                a: t,
                dst: src,
                elements_per_row,
                origin,
            } => {
                let load = matches!(s, Stmt::TileLoad { .. });
                let (atom, role) = plan_of(self, *t)?;
                if !load && role != Role::C {
                    return Err(decline(
                        self.k,
                        format!(
                            "a `TileStore` of operand tile `{}` (only accumulator tiles are stored: an operand has two lanes per element on some atoms)",
                            self.k.allocs[*t as usize].name
                        ),
                    ));
                }
                let (epr, o0, o1) = (
                    self.expr(elements_per_row)?,
                    self.expr(&origin[0])?,
                    self.expr(&origin[1])?,
                );
                self.line(d, "{");
                self.line(
                    d + 1,
                    &format!(
                        "const long long rf_epr = (long long)({epr}), rf_o0 = (long long)({o0}), rf_o1 = (long long)({o1});"
                    ),
                );
                for e in 0..atom.fragment(role).1 {
                    let (r, c) = atom.coord(role, e);
                    let off = format!("(rf_o0 + {r}) * rf_epr + rf_o1 + {c}");
                    let place = self.tile_elem(src, &off)?;
                    let frag = Self::frag(*t, atom, role, e);
                    let line = if load {
                        format!("{frag} = {place};")
                    } else {
                        format!("{place} = {frag};")
                    };
                    self.line(d + 1, &line);
                }
                self.line(d, "}");
            }
            Stmt::TileFill { d: t, value } => {
                let (atom, role) = plan_of(self, *t)?;
                let ek = self.k.allocs[*t as usize].elem;
                let (ty, _) = self.spelled(ek, "a tile fill value")?;
                let v = self.expr(value)?;
                self.line(d, "{");
                self.line(d + 1, &format!("const {ty} rf_v = {v};"));
                for e in 0..atom.fragment(role).1 {
                    let frag = Self::frag(*t, atom, role, e);
                    self.line(d + 1, &format!("{frag} = rf_v;"));
                }
                self.line(d, "}");
            }
            Stmt::TileMma { d: td, a, b, c } => {
                let (atom, _) = plan_of(self, *td)?;
                self.line(
                    d,
                    &format!(
                        "t{td} = {};",
                        atom.call(&format!("t{a}"), &format!("t{b}"), &format!("t{c}"))
                    ),
                );
            }
            other => {
                return Err(decline(self.k, format!("statement `{}`", variant(other))));
            }
        }
        Ok(())
    }
}
