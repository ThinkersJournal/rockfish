# Changelog

## 0.0.3

- New crate `rockfish-core` (unpublished): `LibrarySpec` per-OS names, search order (`ROCKFISH_<NAME>_PATH`, `HIP_PATH`, `ROCM_PATH`, loader), `Library::open` and `Library::symbol` returning `LoadError` instead of panicking.

## 0.0.2

- Docs: `docs/DESIGN.md` (R3) and `docs/SPIKE-R3b-versions.md` (HIP binding differences across releases).

## 0.0.1

- Docs: `docs/SURVEY.md` (R2), the survey of existing ROCm/HIP crates with a Windows HIP runtime measurement. Skeleton crates stay unpublished (`publish = false`).
