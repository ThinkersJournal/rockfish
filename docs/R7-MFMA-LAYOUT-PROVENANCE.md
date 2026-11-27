<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# R7-D15 layout provenance table (calculator only)

TASK from the PM via pm-kernels, 2026-11-27 (CireSnave, board 525: C): calculator first, ISA PDFs only where it is silent.
**Result: the calculator was not silent for any layout fact below, so no ISA PDF was opened and nothing here is ISA-derived.**
(`docs/AMD-ISA-DERIVED-LAYOUTS.md` does not exist.)

Source for every row: `github.com/ROCm/amd_matrix_instruction_calculator` at commit **94b1f16bf3a2c7089f0f3ec0b3eb303396a18776**, MIT, "Copyright (c) 2022-2024 Advanced Micro Devices, Inc." Read through `gh api` raw contents; not cloned, nothing run. sha256: `LICENSE` `d9df59fe...b2f91`, `README.md` `4bd07af8...e4fc`, `matrix_calculator.py` `53b02785...e185d` (full hashes in `docs/DESIGN-R7-MFMA.md` 2a). `py:N` = line N of `matrix_calculator.py` at that commit. Rows were first evaluated by hand from the code (**verified by run: NO**); the PM session then executed the calculator and compared the points listed below, so those cells read YES (see the run record). Rows not named in it stay NO. Lanes: `v` = VGPR, `{lane}` = wavefront lane.

| Atom | Layout fact | Source | Verified by run |
|---|---|---|---|
| `v_mfma_f32_16x16x4f32` (gfx908 CDNA1, gfx90a CDNA2) | m=16, n=16, k=4, blocks=1, in fp32, out fp32 | py:338 (cdna1), py:780 (cdna2) | NO |
| `v_mfma_f64_16x16x4f64` (gfx90a) / `v_mfma_f64_16x16x4_f64` (CDNA3) | m=16, n=16, k=4, blocks=1, fp64 in and out | py:1242 (cdna2), py:1904 (cdna3) | NO |
| CDNA wave | lanes 0-63 only | py:5379 (`lane > 63` is invalid) | NO |
| MFMA A input lane | `elements_in_contiguous_gprs = int(K / int(64/(M*B)))` (=1 for 16x16x4, B=1); `lane = b*M + int(k/elems)*M*B + i`, so **A[i][k] in lane 16k + i** | py:5454-5455, 5470, 5476, 5478 | YES (A[3][2] = v0 lane 35) |
| MFMA B input lane | same function with the outer index `j`: **B[k][j] in lane 16k + j** (BLGP=0) | py:5454-5478 | YES (B[1][5] = v0 lane 21) |
| MFMA input register | one 32-bit register for f32 (`local_element = k % elems = 0`), a register pair for f64 (name from py:3692) | py:5464-5466 region, py:3692 | YES for f64 (A[6][3] = v[1:0] lane 54); f32 one register v0 (A, B above) |
| MFMA f32 D output | `multirow_height=4`, `multirows_per_register=int(64/N)=4`, `blocks_per_register=1`; register `int(i/16)*4 + i%4 = i % 4`; lane `(int(i/4)%4)*N + j`: **D[i][j] in v[i % 4], lane 16*(i div 4) + j** | py:5516-5522, 5527-5533, 5539-5545 | YES (D[6][3] = v2 lane 19; D[13][7] = v1 lane 55) |
| MFMA f64 D output | `multirow_height=1`: **register pair `i div 4`, lane 16*(i % 4) + j** | same lines, data_size 64 branch py:5518 | YES (D[6][3] = v[3:2] lane 35) |
| `v_wmma_f32_16x16x16_f16` (rdna3 gfx11xx; rdna4 gfx12xx has its own row) | m=16, n=16, k=16, blocks=1, fp16 in, fp32 out | py:2302 (rdna3), py:2436 (rdna4) | NO |
| RDNA wave | `wave_width` is an instruction setting; lanes `< wave_width` | py:6205 (`lane >= self.wave_width` is invalid, Gfx11) | NO |
| WMMA A/B input | register named by `_get_reg_name(16, ..., k)`: **VGPR `k div 2`, half `k % 2`**; lanes `i` and `i + 16` at wave32 (four copies, `i`, `i+16`, `i+32`, `i+48`, at wave64) | py:6256, 6262-6267, name helper py:3692 (16-bit branch) | YES (A[3][9] = v4, lanes 3 and 19, bits [31:16]) |
| WMMA f32 D output (wave32) | `rows_per_reg_slot=2`, `skip_half=1`, `regno = int(i/2) + (opsel>>2)`; `rows_per_vgpr=32`; lane `(16*(i%32)+j) % 32`: **D[i][j] in v[i div 2], lane 16*(i % 2) + j** | py:6295-6297, 6303-6304 | YES (D[5][3] = v2 lane 19) |
| WMMA f16 D example | README example for `v_wmma_f16_16x16x16_f16` D: row 0 in v0 lanes 0-15, row 1 in v0 lanes 16-31 (agrees with the formula shape) | README.md:1186 | NO |

**Not covered (UNREAD, no source):** gfx942 instruction rows other than the f64 row above (the f32 CDNA3 row was not located), gfx950/CDNA4, RDNA4 layout code (`InstCalcGfx12`), `blgp`/`cbsz`/`abid` behaviour, wave64 on RDNA, and the *semantic* claim that hardware computes `D = A*B + C` with these layouts (the calculator describes layout, and its own correctness against silicon is not something we checked). When one of these needs a source, the calculator is tried first; an ISA PDF is opened only if it is silent, with the PM told first if its agreement text differs from the MI300 one.

## Run record (PM session, 2026-11-27; relayed by pm-kernels)

AMD's `matrix_calculator.py` at the same pin (94b1f16bf3, sha256 `53b02785...185d`, 340,085 bytes, script unmodified, read via the API into a folder outside every repo, with a 5-line local `tabulate` stand-in on `PYTHONPATH`) was **executed** by the PM session and its output compared with the formulas above. Same artifact, a different method (executing versus reading); it is still the calculator, not silicon: no cell here is a hardware fact. The tool spells the instruction names with an underscore before the type: **`v_mfma_f32_16x16x4_f32`** and **`v_mfma_f64_16x16x4_f64`** (the calculator's table keys read `..16x16x4f32`/`..16x16x4f64` for CDNA1/2 and `..16x16x4_f64` for CDNA3: py:338, 780, 1242, 1904); RDNA3 `v_wmma_f32_16x16x16_f16`.

Check list (all MATCH):

| Instruction | Coordinate | Calculator output |
|---|---|---|
| `v_mfma_f32_16x16x4_f32` | A[3][2] | v0 lane 35 |
| `v_mfma_f32_16x16x4_f32` | B[1][5] | v0 lane 21 |
| `v_mfma_f32_16x16x4_f32` | D[6][3] | v2 lane 19 |
| `v_mfma_f32_16x16x4_f32` | D[13][7] | v1 lane 55 |
| `v_mfma_f64_16x16x4_f64` | D[6][3] | v[3:2] lane 35 |
| `v_mfma_f64_16x16x4_f64` | A[6][3] | v[1:0] lane 54 |
| `v_wmma_f32_16x16x16_f16` | D[5][3] | v2 lane 19 |
| `v_wmma_f32_16x16x16_f16` | A[3][9] | v4 lanes 3 and 19, bits [31:16] |

`tests/mfma_layout_reference.rs` (#63) asserts these eight points as value checks, with wrong-lane mutants. Still UNREAD, unchanged: gfx942 f32 row, gfx950, RDNA4 layout code, `blgp`/`cbsz`/`abid`, wave64. The ISA PDFs stay unopened (nothing was silent).
