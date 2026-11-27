<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# R7-D15/D16/D17: lowering Unpopped's opaque tile to AMD matrix instructions (design only)

Design note; **nothing here is built, and nothing is lowered.** Labels as in `docs/DESIGN.md`: **[measured]**, **[read]**, **[unverified]**. Written 2026-11-27 on pm-kernels' TASK (rows R7-D15..D17 in `ROADMAP.md`).

**Standing text (board 519 B), verbatim as it must travel:** lifted by CireSnave 2026-11-26 (board 519 B) as his risk call; not a clearance; no attorney; the Intel/NVIDIA tile-instruction method and medium claims remain NEEDS MORE as read. AMD matrix patents (PM ODP search + patents-2, as relayed 2026-11-27): 16 outside as read, 3 read-gaps, nothing AMD-shaped shipped; no hold on a design row. **A shipped MFMA/WMMA statement is a chart read first: the atom list in section 4 goes to the PM before any code.**

## 1. What Unpopped 0.62.0 gives us [read]

`Stmt::TileLoad { d, src, elements_per_row, origin }`, `TileMma { d, a, b, c }` (`D = A.B + C`; products exact, ascending `k`, one rounding to the accumulator type; `a`/`b` one element type, `c`/`d` another: `F32`, or `F64` over `F64`), `TileFill`, `TileStore` (one writer), `MemorySpace::Tile` with `AllocDecl::tile: TileShape { rows, cols }`. A tile is held by the whole subgroup, no element is addressable, every statement is reached in uniform control. **No transpose option exists and none is to be added** (US12405770B2 hold). Source: Unpopped `docs/emitter-migration-b1.md` and the doc comments in `kernel.rs`/`exec.rs`, at 0.62.0. The shape is `rows x cols` per tile, so an MFMA's `M x N x K` is **not** a field of the IR: `a` is `m x k`, `b` is `k x n`, and the emitter reads them off the three allocations.

## 2. What AMD's public documentation says (second-hand: see the caveat)

**Caveat on every row below:** the pages were fetched through a summarising model, not as raw text. "READ" means the fact appeared in that returned text. Nothing here was checked on a device or with `hipcc` (no ROCm box **[measured]**). The ISA guides (PDF), the clang builtin docs and `AMDGPUUsage` past its first 100k characters were **not opened**. Sources: (A) `https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html`; (B) `https://github.com/ROCm/amd_matrix_instruction_calculator` README; (C) `https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html`; (D) `https://llvm.org/docs/AMDGPUUsage.html`, first 100k characters.

| Target | Matrix instructions | Status |
|---|---|---|
| gfx908 (CDNA1, MI100) | MFMA | READ (A, B) |
| gfx90a (CDNA2, MI200) | MFMA | READ (A, B) |
| gfx942 (CDNA3, MI300) | matrix instructions (the calculator supports it); instruction list not seen | READ (B), list UNREAD |
| gfx950 (CDNA4) | unknown | UNREAD |
| gfx1100..gfx1153 (RDNA3) | WMMA | READ (B) |
| gfx1200/gfx1201 (RDNA4) | WMMA, sparse SWMMAC | READ (B) |

Wave size: CDNA 64 is **inferred** from lane numbers 0-63 in A and B (neither page states it); RDNA3/4 are wave32 by default with a wave64 feature (D: "When disabled native wavefront size 32 is used, when enabled wavefront size 64 is used"). The existing `wavefront_of` list in `rockfish-hip-emit` is still UNVERIFIED against an AMD page, and this note does not verify it.

**MFMA shapes, CDNA2, from A (A/B in, C/D out), READ:** FP32->FP32 32x32x2, 16x16x4, 4x4x1; FP16->FP32 32x32x8, 16x16x16, 4x4x4; INT8->INT32 same shapes as FP16; BF16->FP32 32x32x8, 16x16x16, 4x4x4 plus 32x32x4, 16x16x8, 4x4x2; FP64->FP64 16x16x4, 4x4x4. Named instructions seen: `v_mfma_f32_16x16x4f32`, `v_mfma_f32_16x16x1f32` (four blocks), `v_mfma_f32_4x4x1f32`, `v_mfma_f64_16x16x4f64`, `v_mfma_f64_4x4x4f64`, each with a `__builtin_amdgcn_mfma_*` spelling given in A for the f32/f64 ones. **RDNA3/4:** `v_wmma_f16_16x16x16_f16`, `v_wmma_f32_16x16x16_f16` (B), `v_swmmac_f32_16x16x32_f16` (RDNA4 sparse, B). `__builtin_amdgcn_wmma_*` spellings were **not read** **[unverified]**.

**Layout, READ:** operands are "distributed over the lanes of the wavefront's vector registers" (A); for 16x16x4 FP32, `A` is row-major and a thread at (x, y) loads `A[x][y]` and `B[y][x]`, each thread holding 4 elements of the 16x16 result (A); on RDNA3 `v_wmma_f16_16x16x16_f16` D row 0 is in v0 lanes 0-15 and row 1 in v0 lanes 16-31 (B). **The exact per-lane row/column formulas for CDNA A, B and C operands were not obtained** and are the gap that decides how `TileLoad` is spelled (section 3).

**Transposition [read, B]:** the calculator's `--transpose` changes only how it prints; hardware modifiers `blgp`, `cbsz`, `abid` broadcast or rotate operands across lanes (A). No hardware transposed-operand option was seen. Consistent with the hold: we expose none.

**Fences (C), READ, near-verbatim:** `__threadfence_block()` "orders memory accesses for all threads within a thread block"; `__threadfence()` "orders memory accesses for all threads on a device"; `__threadfence_system()` the whole system; without a fence or synchronisation HIP does not guarantee one thread's writes are visible to another; `__syncthreads()` includes a threadfence (paraphrase). The separate HIP memory-model page was not opened.

## 2a. The calculator source, read raw (MIT route; added 2026-11-27 after board 525)

**No AMD ISA text was read by this lane, from any source** (PM FIX 2026-11-27: the ISA specification agreement bars use for patent analysis; no ISA PDF, and no summary of one). Section 2's pages (A-D) are the blog, the calculator README, HIP docs and LLVM `AMDGPUUsage`, not an ISA guide.

Source: `github.com/ROCm/amd_matrix_instruction_calculator` at the pinned commit **94b1f16bf3a2c7089f0f3ec0b3eb303396a18776** (`main` when read), fetched with `gh api` raw contents, **not cloned, nothing run**. Licence MIT, "Copyright (c) 2022-2024 Advanced Micro Devices, Inc."; it is kept as a notice here, and no calculator text or code is copied into Rockfish. Files read (sha256):

| File | Bytes | sha256 |
|---|---|---|
| `LICENSE` | 1098 | `d9df59fe2eb53910bf6d1c97a316b7510f1356946ee110dbb54e9d40508b2f91` |
| `README.md` | 147878 | `4bd07af8340a28c95ce74a93becebc0735866bd5646acb3789ff7018c67fe4fc` |
| `matrix_calculator.py` | 340085 | `53b027855ca44120401eeff69f41961821d1a393b163e112f7aa4d2a313e185d` |

**Hand-evaluated from the code, not executed [read, derived]:** `class InstCalcGfx9` (CDNA, lanes 0-63 only: `check_valid_reg_lane`, line ~5362), `__get_input_reg_lanes` (line 5401) and `__get_output_reg_lanes` (after it), with `M` the outer dimension, `K` the inner, `B` the blocks:

- Input, `elements_in_contiguous_gprs = int(K / int(64 / (M*B)))`; `lane = b*M + int(k / elements_in_contiguous_gprs)*M*B + i`. For **`v_mfma_f32_16x16x4f32` and `v_mfma_f64_16x16x4f64`** (M=16, K=4, B=1): contiguous = 1, so **A[i][k] is in lane `16*k + i`, B[k][j] in lane `16*k + j`**, one register (a pair for f64). These are the code's formulas, evaluated by hand; the instruction's own M/K/B values were taken from the README's instruction list (lines 177, 198), not from the table in the `.py`. UNVERIFIED against a run.
- Output, 32-bit data: `multirow_height = 4`, `multirows_per_register = int(64/N)`, register `int(i/(4*mr))*4 + i%4`, lane `(int(i/4) % mr)*blocks_per_register*N + j`. For f32 16x16x4: **D[i][j] in register `i % 4`, lane `16*(i div 4) + j`**. 64-bit data: `multirow_height = 1`, so for f64 16x16x4: **register pair `i div 4`, lane `16*(i % 4) + j`**. (Hand-evaluated; UNVERIFIED against a run.)
- Cross-check against the blog read in section 2 (a different source, but both are AMD's; one method is not independent): the blog's "each thread holds 4 elements of the 16x16 output" agrees with 4 registers per lane; its "thread at (x, y)" wording is not compared.
- `class InstCalcGfx11` (RDNA3): the wave width is an instruction setting (`wave_width`, 32 or 64); input element pairs are stored 16 lanes apart (lane `i` and `i + 16` at wave32), output lane `(N*(i % rows_per_vgpr) + j) % wave_width` with `rows_per_vgpr = int(wave_width*16/N)`. For `v_wmma_f32_16x16x16_f16` at wave32 that puts D row `i` in lane `(16*i + j) % 32`. **The output register index (lines just before 6307) and the input register index were not read.** The README example for `v_wmma_f16_16x16x16_f16` D (line 1186) agrees on rows 0 and 1 (v0 lanes 0-15 and 16-31).
- **Not read:** CDNA3 (gfx942) and gfx950 specifics (the class is shared, but the instruction table rows were not read), `blgp`/`cbsz`/`abid` handling beyond the code comments, RDNA4. Anything not above is UNVERIFIED.

**What this changes:** the section 2 layout statements sourced from the summarised blog stay UNVERIFIED; the formulas above are the only layout claims with a raw source, and they are derived by hand.

## 3. Mapping, and why it is not one-to-one

Unpopped's tile is opaque (no lane mapping); AMD's MFMA/WMMA is **per-lane fragments in VGPRs** with a layout the hardware fixes. So a tile lowers to a **fragment register group plus the fixed AMD layout**, and the emitter owns the load/store code that moves between memory and that layout:

- `MemorySpace::Tile` allocation -> a per-lane fragment variable (a vector type sized by the instruction's operand width).
- `TileLoad`/`TileStore` -> the per-lane loads/stores that realise the layout, with every lane's address computed from `elements_per_row` and `origin`. Needs the layout formulas (gap above). Not the `__builtin_amdgcn_*` load forms; ordinary loads.
- `TileFill` -> fill every fragment slot.
- `TileMma` -> the one builtin whose `M x N x K` and types equal the three allocations' shapes and element types **exactly**; no match, a decline.

Precision: the oracle's contract is exact products, ascending `k`, one rounding; hardware order and width differ, so a device result is compared within a tolerance, never by bytes (Unpopped's own rule). The host-shim differential used for the kernel layer cannot test this: the shim would model our own reading of AMD's layout (the same lane model on both sides is one measurement). **A device run is rig-only** (R5-H); until then every matrix lowering would be UNMEASURED.

## 4. Proposed first atoms (a proposal; the PM reads a chart before any ships)

| Atom | Target | Shape | In -> out | Why it is first |
|---|---|---|---|---|
| `v_mfma_f32_16x16x4f32` | gfx908/90a/942 | 16x16x4 | f32 -> f32 | layout example given in A; closest to the oracle (no narrowing) |
| `v_mfma_f64_16x16x4f64` | gfx90a/942 | 16x16x4 | f64 -> f64 | matches `TileMma`'s `F64` over `F64` |
| `v_wmma_f32_16x16x16_f16` | gfx11xx | 16x16x16 | f16 -> f32 | RDNA is the laptop's family (gfx1036 is **not** a WMMA part as far as read: UNREAD) |

Nothing else. gfx950/CDNA4, sparse, int8, bf16, FP8 and every `blgp`/`cbsz`/`abid` form stay declined by name.

## 5. What the emitter declines by name today and after

Today (0.62.0 take): all four statements and `MemorySpace::Tile` ("Tile" in the message), unchanged by this note. When an atom ships: any tile shape or type pair not in the atom list, any target not named, a Tile statement in a kernel with no subgroup size stated, `Wgmma`/`WgmmaCommit`/`Tcgen05Mma` (R7-D15/16/17 stay declined: they are NVIDIA warpgroup and tensor-memory statements, not tiles; AMD has no counterpart read).

## 6. Order of work and what blocks each

1. **Send the PM the atom list in section 4** for the chart read. Blocks code.
2. Read the per-lane layouts: run the calculator `-R/-M` for the atom on gfx90a and gfx11xx, or read A's figures and the ISA guides. Blocks `TileLoad`/`TileStore`.
3. Text lowering of one atom, host-compile of the surrounding code only; the MFMA builtin itself needs a ROCm compiler. UNMEASURED.
4. Device run on the rig (R5-H, second AMD machine).

— lane: rockfish
