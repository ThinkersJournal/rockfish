//! Value-level run of the emitted integer `Div`, `Shl` and `Shr` kernels on the host.
//!
//! Each kernel is emitted for the integer dtypes, compiled with `clang++` against the shim
//! `tests/shim/hip/hip_runtime.h` (the grid variables are plain globals there, so one "thread" runs
//! the grid-stride loop over the whole input), run on boundary vectors and compared with Rust's own
//! integer operations, which implement the same ruled rule independently of the emitted guards
//! (Unpopped `docs/int-div-shift-semantics.md`): a zero divisor gives the dividend, signed
//! `MIN / -1` gives `MIN`, and a shift count is taken modulo the width. The operand pairs are
//! asymmetric (every `a` against every `b`, not `a == b`), so swapped operands cannot pass.
//!
//! Mutation control (run by hand, then reverted): dropping the signed `MIN / -1` guard crashes
//! `Div_i32` (x86 `idiv` traps), and making signed `Shr` logical fails 85 of 286 `Shr_i32` pairs.
//! **Not observable here:** the `& (N - 1)` mask on the count. x86 hardware masks a 32/64-bit shift
//! count itself, so a wrong mask (tried: 63 at i32) still passes on this host; the count cases are
//! checked but the mask is only proven by reading, until a device runs it (rig row).
//!
//! **What this shows:** the emitted guards give the ruled results under a host C++ compiler. **What
//! it does not:** hipcc, the device, or GPU codegen of the same text (rig rows stand). Skips with a
//! printed message when no `clang++` is found (`ROCKFISH_CLANG` names one). With `REQUIRE_CLANG=1` it fails instead.

use std::path::PathBuf;
mod common;

use std::process::Command;

use rockfish_hip_emit::Hip;
use unpopped::ir::{BinaryOp, OpDef, input};
use unpopped::try_generate;
use unpopped_vocab::{ElementKind, OpCategory, OperandDesc, TargetId, structure_key};

fn clang() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ROCKFISH_CLANG") {
        return Some(PathBuf::from(p));
    }
    [
        PathBuf::from("clang++"),
        PathBuf::from(
            "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin/clang++.exe",
        ),
    ]
    .into_iter()
    .find(|c| {
        Command::new(c)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    })
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Div,
    Shl,
    Shr,
}

fn lower(op: Op, dt: ElementKind) -> (String, String) {
    let t = TargetId::parse("rocm:gfx906").unwrap();
    let (name, body) = match op {
        Op::Div => ("div", input(0) / input(1)),
        Op::Shl => ("shl", input(0).binary(BinaryOp::Shl, input(1))),
        Op::Shr => ("shr", input(0).binary(BinaryOp::Shr, input(1))),
    };
    let o = OpDef::elementwise(name, 2, &[dt], body);
    let d = OperandDesc::new(1, &[1 << 20], &[1], dt, 4);
    let key = structure_key(OpCategory::BinaryElementwise, &[d, d, d], t);
    let src = try_generate(&o, &key, &Hip).expect("generate").source;
    let tag = format!("{dt:?}").to_lowercase();
    (src, format!("rockfish_hip_{name}_{tag}"))
}

/// Compile `src` with a `main` that runs `kernel` over the pairs `(a[i], b[i])` and prints one
/// result per line; return those lines.
fn run(
    cc: &PathBuf,
    label: &str,
    src: &str,
    kernel: &str,
    (cty, fmt): (&str, &str),
    a: &[String],
    b: &[String],
) -> Vec<String> {
    let n = a.len();
    let main = format!(
        r#"
#include <cstdio>
int main() {{
    static const {cty} A[] = {{{av}}};
    static const {cty} B[] = {{{bv}}};
    static {cty} O[{n}];
    blockIdx.x = 0; blockDim.x = 1; threadIdx.x = 0; gridDim.x = 1;
    {kernel}(A, B, O, {n}LL);
    for (int i = 0; i < {n}; ++i) std::printf("{fmt}\n", ({cast})O[i]);
    return 0;
}}
"#,
        av = a.join(","),
        bv = b.join(","),
        cast = if fmt.contains("llu") {
            "unsigned long long"
        } else {
            "long long"
        },
    );
    let dir = std::env::temp_dir().join("rockfish-intrun");
    std::fs::create_dir_all(&dir).unwrap();
    let cpp = dir.join(format!("{label}.cpp"));
    let exe = dir.join(format!("{label}{}", std::env::consts::EXE_SUFFIX));
    std::fs::write(&cpp, format!("{src}\n{main}")).unwrap();
    let shim = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/shim");
    let out = Command::new(cc)
        .args([
            "-std=c++17",
            "-x",
            "c++",
            "-O0",
            "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
        ])
        .arg(format!("-I{}", shim.display()))
        .arg(&cpp)
        .arg("-o")
        .arg(&exe)
        .output()
        .expect("clang runs");
    assert!(
        out.status.success(),
        "{label} does not build:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run = Command::new(&exe).output().expect("exe runs");
    assert!(run.status.success(), "{label} crashed");
    String::from_utf8(run.stdout)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect()
}

macro_rules! check_type {
    ($cc:expr, $dt:expr, $t:ty, $ut:ty, $cty:expr, $fmt:expr, $signed:expr, $tag:expr) => {{
        let mut vals: Vec<$t> = vec![0, 1, 2, 3, 7, 100, <$t>::MAX, <$t>::MIN, <$t>::MAX - 1, <$t>::MIN.wrapping_add(1)];
        if $signed {
            vals.extend([(-1i64) as $t, (-2i64) as $t, (-7i64) as $t]);
        }
        let bits = <$t>::BITS as i64;
        let counts: Vec<i64> = vec![0, 1, 2, bits - 1, bits, bits + 1, 2 * bits - 1, 2 * bits, 2 * bits + 3];
        // Pairs: every value against every value and every count (counts as the type).
        let mut a: Vec<$t> = Vec::new();
        let mut b: Vec<$t> = Vec::new();
        for &x in &vals {
            for &y in vals.iter().chain(counts.iter().map(|c| c).map(|c| Box::leak(Box::new(*c as $t)) as &$t)) {
                a.push(x);
                b.push(y);
            }
        }
        let lit = |v: $t| -> String {
            // MIN has no literal form in C++; spell it as (MIN+1)-1 through a cast of the bit pattern.
            if $signed && v == <$t>::MIN {
                format!("({}){}", $cty, (<$t>::MIN as i128 + 1))
                    .replace(&format!("({})", $cty), "")
                    + &format!("-1")
            } else {
                format!("{}{}", v, if bits == 64 { if $signed { "LL" } else { "ULL" } } else if $signed { "" } else { "u" })
            }
        };
        let la: Vec<String> = a.iter().map(|&v| lit(v)).collect();
        let lb: Vec<String> = b.iter().map(|&v| lit(v)).collect();
        for op in [Op::Div, Op::Shl, Op::Shr] {
            let (src, kernel) = lower(op, $dt);
            let label = format!("{op:?}_{}", $tag);
            let got = run($cc, &label, &src, &kernel, ($cty, $fmt), &la, &lb);
            assert_eq!(got.len(), a.len(), "{label}: wrong number of results");
            let mut bad = Vec::new();
            for i in 0..a.len() {
                let (x, y) = (a[i], b[i]);
                let want: $t = match op {
                    Op::Div => if y == 0 { x } else { x.wrapping_div(y) },
                    Op::Shl => x.wrapping_shl((y as $ut & (<$t>::BITS - 1) as $ut) as u32),
                    Op::Shr => x.wrapping_shr((y as $ut & (<$t>::BITS - 1) as $ut) as u32),
                };
                if got[i] != format!("{want}") {
                    bad.push(format!("{op:?}({x}, {y}): kernel {} want {want}", got[i]));
                }
            }
            assert!(bad.is_empty(), "{label}: {} of {} boundary results differ:\n{}", bad.len(), a.len(), bad.iter().take(12).cloned().collect::<Vec<_>>().join("\n"));
            eprintln!("{label}: {} boundary pairs match the ruled rule", a.len());
        }
    }};
}

#[test]
fn emitted_integer_div_and_shifts_give_the_ruled_results() {
    let Some(cc) = clang() else {
        common::skip_or_fail(
            "no clang++ found: the host value-level run did not run (set ROCKFISH_CLANG)",
        );
        return;
    };
    check_type!(&cc, ElementKind::I32, i32, u32, "int", "%lld", true, "i32");
    check_type!(
        &cc,
        ElementKind::I64,
        i64,
        u64,
        "long long",
        "%lld",
        true,
        "i64"
    );
    check_type!(
        &cc,
        ElementKind::U32,
        u32,
        u32,
        "unsigned int",
        "%llu",
        false,
        "u32"
    );
    check_type!(
        &cc,
        ElementKind::U64,
        u64,
        u64,
        "unsigned long long",
        "%llu",
        false,
        "u64"
    );
}
