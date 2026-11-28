// SPDX-License-Identifier: MIT OR Apache-2.0
//! R7-D15: the emitted tile kernels (`tests/fixtures/tile_*.cpp`, which `tests/tile_lowering.rs` pins to what `Hip` emits)
//! compiled by clang's AMDGPU target to assembly (`-S`, never run) for the right `-mcpu`, and the assembly names the atom's
//! instruction. Standing text (board 519 B): lifted by CireSnave 2026-11-26 (board 519 B) as his risk call; not a clearance;
//! no attorney; the Intel/NVIDIA tile-instruction method and medium claims remain NEEDS MORE as read.
//!
//! **What this shows:** clang accepts the three builtin spellings with the fragment types, and selects `v_mfma_f32_16x16x4*`
//! and `v_mfma_f64_16x16x4*` on gfx90a/gfx942 (gfx908 for the f32 atom) and `v_wmma_f32_16x16x16_f16` on gfx1100.
//! **What it does not show:** that the kernel computes anything: nothing runs. This machine's Radeon 610M (gfx1036) has neither
//! MFMA nor WMMA and there is no ROCm box: device run UNMEASURED. A box whose clang lacks the AMDGPU target skips with a printed
//! message (`ROCKFISH_AMDGPU_CLANG` names one); `REQUIRE_AMDGPU_CLANG=1` makes that a failure.

use std::path::PathBuf;
use std::process::Command;

fn amdgpu_clang() -> Option<PathBuf> {
    let candidates: Vec<PathBuf> = match std::env::var("ROCKFISH_AMDGPU_CLANG") {
        Ok(p) if p.is_empty() => vec![],
        Ok(p) => vec![PathBuf::from(p)],
        Err(_) => [
            "clang++",
            "clang++-20",
            "clang++-19",
            "clang++-18",
            "clang++-17",
            "clang++-16",
        ]
        .into_iter()
        .map(PathBuf::from)
        .collect(),
    };
    candidates.into_iter().find(|c| {
        Command::new(c)
            .arg("--print-targets")
            .output()
            .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains("amdgcn"))
    })
}

fn compile(cc: &PathBuf, fixture: &str, mcpu: &str) -> Result<String, String> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = Command::new(cc)
        .args([
            "--target=amdgcn-amd-amdhsa",
            &format!("-mcpu={mcpu}"),
            "-nogpulib",
            "-x",
            "c++",
            "-std=c++17",
            "-S",
            "-o",
            "-",
        ])
        .arg(format!(
            "-I{}",
            manifest.join("tests/shim_amdgcn").display()
        ))
        .arg(manifest.join("tests/fixtures").join(fixture))
        .output()
        .expect("clang runs");
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).into_owned())
    }
}

fn each(cases: &[(&str, &str, &str)]) {
    let Some(cc) = amdgpu_clang() else {
        let msg = "no clang with the AMDGPU target found (set ROCKFISH_AMDGPU_CLANG); tile syntax check skipped";
        assert!(std::env::var("REQUIRE_AMDGPU_CLANG").is_err(), "{msg}");
        eprintln!("SKIP: {msg}");
        return;
    };
    for (fixture, mcpu, mnemonic) in cases {
        let asm =
            compile(&cc, fixture, mcpu).unwrap_or_else(|e| panic!("{fixture} on {mcpu}:\n{e}"));
        assert!(
            asm.contains(mnemonic),
            "{fixture} on {mcpu}: no `{mnemonic}` in the assembly"
        );
    }
}

#[test]
fn mfma_f32_compiles_and_selects_the_instruction() {
    each(&[
        ("tile_mfma_f32.cpp", "gfx908", "v_mfma_f32_16x16x4f32"),
        ("tile_mfma_f32.cpp", "gfx90a", "v_mfma_f32_16x16x4f32"),
        ("tile_mfma_f32.cpp", "gfx942", "v_mfma_f32_16x16x4_f32"),
    ]);
}

#[test]
fn mfma_f64_compiles_and_selects_the_instruction() {
    each(&[
        ("tile_mfma_f64.cpp", "gfx90a", "v_mfma_f64_16x16x4f64"),
        ("tile_mfma_f64.cpp", "gfx942", "v_mfma_f64_16x16x4_f64"),
    ]);
}

#[test]
fn wmma_f16_compiles_and_selects_the_instruction() {
    each(&[("tile_wmma_f16.cpp", "gfx1100", "v_wmma_f32_16x16x16_f16")]);
}

/// Control: the same text for a target without the instruction does not compile, so a pass above is the atom's.
#[test]
fn the_wrong_target_is_refused_by_clang() {
    let Some(cc) = amdgpu_clang() else {
        eprintln!("SKIP: no AMDGPU clang");
        return;
    };
    assert!(
        compile(&cc, "tile_mfma_f32.cpp", "gfx906").is_err(),
        "gfx906 has no MFMA"
    );
    assert!(
        compile(&cc, "tile_wmma_f16.cpp", "gfx1030").is_err(),
        "gfx1030 has no WMMA"
    );
}
