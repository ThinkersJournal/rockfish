// SPDX-License-Identifier: MIT OR Apache-2.0
//! The kernel-layer lowerer, slice S1 (CireSnave board 506 plan; pm-kernels 2026-11-26): the
//! skeleton. A `KernelFn` is lowered to one HIP `__global__` function.
//!
//! **Spelled:** parameters (buffers and scalars), locals, `Let`, `Load`, `Assign`, `Store`,
//! `If`, `For`, `While`, `Break`, `Continue`, `Return`, and the expressions `Const`, `ConstInt`,
//! `Local`, `Exec` (thread, block, block-dim and grid-dim ids), `Add`, `Sub`, `Mul`, `Div` at
//! a float type, and the six comparisons `CmpEq`..`CmpGe` (`(a < b) ? 1 : 0`, in the operand
//! dtype, NaN as the C operators; Unpopped's `BinaryOp` doc), at `f32`, `f64`, `i32`, `i64`, `u32` and `u64`. **Everything else is declined by
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

use std::collections::BTreeMap;
use std::fmt::Write as _;

use unpopped::backend::{GeneratedKernel, LowerError};
use unpopped::exec::ExecId;
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{BufferRole, KernelFn, ParamKind, Place, Stmt};
use unpopped::kernel_interface::KernelInterface;
use unpopped_vocab::ElementKind;

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
            ScalarExpr::Add(a, _)
            | ScalarExpr::Sub(a, _)
            | ScalarExpr::Mul(a, _)
            | ScalarExpr::Div(a, _) => self.ty_of(a),
            other => Err(decline(self.k, format!("expression `{}`", variant(other)))),
        }
    }

    fn spelled(
        &self,
        ek: ElementKind,
        what: &str,
    ) -> Result<(&'static str, Option<&'static str>), LowerError> {
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
            other => Err(decline(self.k, format!("expression `{}`", variant(other)))),
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
            other => Err(decline(self.k, format!("place `{}`", variant(other)))),
        }
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
            Stmt::Break => self.line(d, "break;"),
            Stmt::Continue => self.line(d, "continue;"),
            Stmt::Return => self.line(d, "return;"),
            other => return Err(decline(self.k, format!("statement `{}`", variant(other)))),
        }
        Ok(())
    }
}

/// Lower one kernel-layer function (S1).
pub(crate) fn lower(kernel: &KernelFn) -> Result<GeneratedKernel, LowerError> {
    if !kernel.allocs.is_empty() {
        return Err(decline(
            kernel,
            "allocations (shared/private/constant arrays) are not lowered",
        ));
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
    };
    cx.block(&kernel.body, 1)?;
    let body = std::mem::take(&mut cx.out);

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
    let _ = writeln!(s, "extern \"C\" __global__ void {name}(");
    let _ = writeln!(s, "    {}", sig.join(",\n    "));
    s.push_str(") {\n");
    s.push_str(&body);
    s.push_str("}\n");
    Ok(GeneratedKernel::new(name.clone(), s)
        .with_interface(KernelInterface::of(kernel).with_entry_symbol(name)))
}
