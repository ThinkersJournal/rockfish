# R3b spike: how much do HIP bindings differ between releases?

Question from `docs/DESIGN.md` section 3: can one set of bindings serve many HIP releases, with symbols resolved per call, or must struct layouts be versioned? **[read]**: compared the 11 bindgen files that `cubecl-hip-sys` 7.14.6085001 ships (HIP builds 41134, 42131, 42133, 42134, 43482, 43483, 43484, 51831, 52802, 53211, 60850), oldest against newest, by parsing struct/union fields and `hip*` function signatures from the generated Rust. Throwaway Python, not committed. Nothing was built or run against a real HIP; I did not map build numbers to ROCm versions.

## Findings

- **`hipDeviceProp_tR0600`**: 107 fields in all 11 files, no field added, removed or retyped between any adjacent pair. The device-properties struct is stable across this whole range.
- Oldest to newest: 182 to 216 structs/unions (38 added, 4 removed), 459 to 546 `hip*` functions (88 added, 1 removed: `hipMemcpy_spt`).
- Functions whose signature text changed (11 by the parser; checked by eye for five): the changes are **ABI-neutral**: `*mut` to `*const` on inputs (`hiprtcCreateProgram` headers, `hipMemcpyHtoD` src) and `c_int` to `c_uint` (`hipCtxGetApiVersion`). Others on the list were parser noise from neighbouring doc comments, not checked individually.
- Struct whose **layout** changed: `hipMemAllocationProp` (virtual-memory management API): a union inserted after `type_`, and the `allocFlags` field's type renamed. Also `hipLaunchAttributeValue`. Neither is on the path of the first wave (devices, streams, memory, modules, hipRTC).

## Consequences for the design

1. Wave 1 can be **bound once from the newest headers** and resolved per symbol at run time; version-specific struct tables are not needed for the properties struct. DESIGN section 3 is corrected accordingly.
2. New symbols (88 of them) are `SymbolMissing`-able per call: the per-call `Result` design already covers them.
3. Rare structs with layout changes (VMM, launch attributes) are fenced behind a runtime version check and added only when a consumer needs them.
4. **Not covered**: releases before this range (ROCm 5.x and earlier use the unversioned `hipDeviceProp_t` and `hipGetDeviceProperties`), Windows HIP SDK builds (the laptop's driver runtime is 6.4.50501 **[measured]** and exports `hipGetDevicePropertiesR0600`), and the math libraries, whose ABI changes (rocBLAS 3 to 4 soname bump) are a separate question for wave 2.
