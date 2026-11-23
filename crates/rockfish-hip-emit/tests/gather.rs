//! The gather (`ReadIndex::Indexed`) on the strided elementwise cell: text checks of the stated ABI
//! and a host run (clang++ against the shim) of the three out-of-range policies.
//!
//! ABI (`baracuda-cuda-emit`'s names): the index operand's pointer takes the index type
//! (`int` / `long long`), `long long gext` (the data extent along the gathered axis) sits between
//! `so_*` and the offsets, and the index dtype rides the kernel name (`_i32` / `_i64`).
//!
//! **What the run shows:** the emitted text, under a host C++ compiler, reads the rows the indices
//! name, leaves (Skip), zeroes (ZeroFill) or clamps (Clamp) out-of-range and negative ones, and never
//! reads outside the data. **What it does not:** hipcc, a device, or any GPU codegen. Skips with a
//! printed message when no `clang++` is found (`ROCKFISH_CLANG` names one). With `REQUIRE_CLANG=1` it fails instead.

use std::path::PathBuf;
mod common;

use std::process::Command;

use rockfish_hip_emit::Hip;
use unpopped::ir::{OobPolicy, OpDef};
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

const DATA: [i64; 2] = [4, 3];
const OUT: [i64; 2] = [5, 3];
/// Row indices (a 1-D index along axis 0: an embedding lookup), with a negative and two past the end.
const IDS: [i64; 5] = [2, 0, -1, 7, 3];

fn source(dt: ElementKind, idt: ElementKind, oob: OobPolicy, name: &str) -> String {
    let t = TargetId::parse("rocm:gfx906").unwrap();
    let o = OpDef::gather(name, &[dt], 0, oob, idt);
    let data = OperandDesc::new(2, &DATA, &[3, 1], dt, 4);
    let idx = OperandDesc::new(2, &OUT, &[1, 0], idt, 4);
    let out = OperandDesc::new(2, &OUT, &[3, 1], dt, 4);
    let key = structure_key(OpCategory::BinaryElementwise, &[data, idx, out], t);
    try_generate(&o, &key, &Hip)
        .unwrap_or_else(|e| panic!("{name}: {e:?}"))
        .source
}

#[test]
fn the_gather_abi_is_the_stated_one() {
    let s = source(ElementKind::F32, ElementKind::I32, OobPolicy::Skip, "g");
    assert!(s.contains("void rockfish_hip_g_f32_i32("), "{s}");
    assert!(
        s.contains("const float* in0,\n    const int* in1,\n    float* out,"),
        "{s}"
    );
    assert!(
        s.contains("    long long so_1,\n    long long gext,\n    long long n)"),
        "gext sits between so_* and n:\n{s}"
    );
    assert!(
        s.contains("long long gidx_raw = (long long)in1[gidx_off];"),
        "{s}"
    );
    assert!(
        s.contains("gidx_raw < 0 ? 0 : (gidx_raw >= gext ? gext - 1 : gidx_raw)"),
        "{s}"
    );
    assert!(s.contains("if (!goob) out[oo] ="), "{s}");
    let s = source(ElementKind::I32, ElementKind::I64, OobPolicy::ZeroFill, "z");
    assert!(s.contains("rockfish_hip_z_i32_i64("), "{s}");
    assert!(s.contains("const long long* in1,"), "{s}");
    assert!(s.contains("out[oo] = goob ? (0) : ("), "{s}");
    // Clamp has no `goob` at all: the store is unconditional.
    let s = source(ElementKind::F64, ElementKind::I32, OobPolicy::Clamp, "c");
    assert!(!s.contains("goob"), "{s}");
}

fn run(cc: &PathBuf, label: &str, src: &str, kernel: &str, cty: &str, ictype: &str) -> Vec<f64> {
    let main = format!(
        r#"
#include <cstdio>
int main() {{
    static {cty} D[12];
    for (int i = 0; i < 12; ++i) D[i] = ({cty})((i / 3) * 10 + (i % 3) + 1);
    static {ictype} I[5] = {{{ids}}};
    static {cty} O[15];
    for (int i = 0; i < 15; ++i) O[i] = ({cty})(-99);
    blockIdx.x = 0; blockDim.x = 1; threadIdx.x = 0; gridDim.x = 1;
    // shape0 shape1 | s0_0 s0_1 | s1_0 s1_1 | so_0 so_1 | gext | n
    {kernel}(D, I, O, 5LL, 3LL, 3LL, 1LL, 1LL, 0LL, 3LL, 1LL, 4LL, 15LL);
    for (int i = 0; i < 15; ++i) std::printf("%.1f\n", (double)O[i]);
    return 0;
}}
"#,
        ids = IDS.map(|v| v.to_string()).join(","),
    );
    let dir = std::env::temp_dir().join("rockfish-gather");
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
        .map(|l| l.parse().unwrap())
        .collect()
}

#[test]
fn out_of_range_indices_follow_the_policy_on_a_host_run() {
    let Some(cc) = clang() else {
        common::skip_or_fail(
            "no clang++ found: the host gather run did not run (set ROCKFISH_CLANG)",
        );
        return;
    };
    let cases = [
        (
            ElementKind::F32,
            "float",
            ElementKind::I32,
            "int",
            OobPolicy::Skip,
        ),
        (
            ElementKind::F32,
            "float",
            ElementKind::I32,
            "int",
            OobPolicy::ZeroFill,
        ),
        (
            ElementKind::F32,
            "float",
            ElementKind::I32,
            "int",
            OobPolicy::Clamp,
        ),
        (
            ElementKind::I32,
            "int",
            ElementKind::I64,
            "long long",
            OobPolicy::ZeroFill,
        ),
        (
            ElementKind::F64,
            "double",
            ElementKind::I64,
            "long long",
            OobPolicy::Skip,
        ),
        (
            ElementKind::U64,
            "unsigned long long",
            ElementKind::I32,
            "int",
            OobPolicy::Clamp,
        ),
    ];
    for (dt, cty, idt, ictype, oob) in cases {
        let tag = format!("{dt:?}").to_lowercase();
        let itag = format!("{idt:?}").to_lowercase();
        let name = format!("g{oob:?}").to_lowercase();
        let src = source(dt, idt, oob, &name);
        let label = format!("{name}_{tag}_{itag}");
        let kernel = format!("rockfish_hip_{name}_{tag}_{itag}");
        let got = run(&cc, &label, &src, &kernel, cty, ictype);
        let mut want = Vec::new();
        for &id in &IDS {
            let oobi = id < 0 || id >= DATA[0];
            for c in 0..3i64 {
                let row = id.clamp(0, DATA[0] - 1);
                let v = (row * 10 + c + 1) as f64;
                want.push(match (oobi, oob) {
                    (false, _) => v,
                    (true, OobPolicy::Skip) => -99.0,
                    (true, OobPolicy::ZeroFill) => 0.0,
                    (true, OobPolicy::Clamp) => v,
                });
            }
        }
        assert_eq!(got, want, "{label} ({oob:?}) differs");
        eprintln!(
            "{label}: matches ({oob:?}, ids {IDS:?}, data rows {})",
            DATA[0]
        );
    }
}
