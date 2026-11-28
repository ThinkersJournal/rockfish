# AMD-ISA-DERIVED facts (rockfish lane only)

Facts here come from **AMD's instruction-set reference**, opened by the Rockfish lane only (board 525 C,
2026-11-27; `C:/amd-docs/rockfish-only/`). They are used as a reference for designing an emitter that
interfaces with AMD products (the document's Specification Agreement, clause 1). Nothing is copied: each fact
is in our own words with document and page. **Patent lanes and the PM do not read this file or use it for any
patent analysis (clause 3).** Agreement clauses 1-3 were read before opening the document and match
`README-WALL.txt`.

Document: *AMD Instinct MI300 Instruction Set Architecture Reference Guide*, cover date 5 August 2025, 553
pages, sha256 `0cec4237...a6e2a` (`PROVENANCE.txt`). Page numbers are the printed "N of 553" footers.

## Does AMD need a commit after MFMA? (R7-D16)

- There is no commit or group-wait instruction for matrix operations. The hardware does not check the
  dependencies in the wait-state tables; software must insert `s_nop` or independent instructions (p. 20,
  section 4.5; pp. 40-41, section 7.1, which says results are not produced in one cycle and may be observable
  early).
- Section 7.5 (pp. 55-58) lists, per first/second instruction pair, the independent instructions required
  between an MFMA and (a) a later read or write of its destination registers (RAW and WAW), (b) a later
  matrix op reading it as SrcA/SrcB or as the C accumulator, (c) a memory or export instruction reading the
  overlapping registers, (d) a vector write over its still-unread C input (WAR). The count depends on the
  first MFMA's pass count (a pass is 4 clock cycles; 2, 4, 8 and 16 passes occur) and on the kind of the
  second instruction. Back-to-back MFMAs of the same shape accumulating into the same destination need 0
  waits (internal forwarding); an overlapping but different layout needs the longer counts; the F64 MFMAs
  have their own rows.
- **Exact wait counts: UNREAD.** Text extraction interleaves the table's columns, so no number is taken from
  it here; a clean read of pp. 55-58 is needed before a lowering inserts any count.

## Is there a tensor-memory analogue? (R7-D17)

- No. Searching the whole text of the document for "tensor memory", "tensor-memory" and "tmem" finds 0 hits;
  the same search method finds "MFMA" 247 times and "AccVGPR" (positive controls). MFMA inputs and outputs
  live in ordinary VGPRs or in the separate accumulation VGPRs (AccVGPRs, moved to and from with
  `V_ACCVGPR_*`; p. 41, with the register file listed on p. 8). The A/B register file is chosen by the two
  bits of the instruction's ACC field and C/D by ACC_CD.
- Scope: MI300 (CDNA3) only. CDNA4/gfx950 and RDNA3/RDNA4 documents are not read (RDNA files pending).

— lane: rockfish
