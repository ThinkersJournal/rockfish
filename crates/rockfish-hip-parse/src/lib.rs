//! The HIP C++ [`unpopped::convert::Frontend`]: the parse side (HIP source -> IR) of Rockfish's
//! HIP C++ ownership (see `docs/DESIGN-R7.md`).
//!
//! Shape follows Baracuda's `baracuda-cuda-parse` (read for structure only; nothing copied): a
//! grammar binding ([`parse_hip`]), a residue list ([`HIP_RESIDUE`]) and a [`HIP`] descriptor. The
//! shared walkers stay in `unpopped::convert`.
//!
//! **Grammar.** No HIP grammar exists on crates.io. This crate binds `tree-sitter-cuda` 0.21,
//! which `tests/convert.rs` shows parses the HIP kernel idioms used there (the corpus is
//! hand-written in that file, and `corpus_parses_without_error_nodes` states which constructs
//! were checked). Anything else is unmeasured.
//!
//! **Not a correctness proof.** The tests are golden vectors for these idioms. A green suite is
//! evidence about them, not about HIP parsing in general. Nothing here compiles or runs HIP.

use tree_sitter::{Parser, Tree};
use unpopped::convert::Frontend;

/// Parse HIP C++ source into a tree-sitter CST (error-tolerant: constructs the grammar does not
/// know become `ERROR` nodes rather than failing). `None` only if the grammar cannot be loaded or
/// the parse is cancelled.
pub fn parse_hip(src: &str) -> Option<Tree> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_cuda::LANGUAGE.into())
        .ok()?;
    parser.parse(src, None)
}

/// HIP constructs the neutral IR does not hold. Their presence (as a substring, checked before
/// parsing) refuses the kernel as inexpressible: shared memory, atomics, barriers and fences,
/// wavefront/lane intrinsics, inline assembly and AMDGCN builtins, library calls, and host-side
/// launch syntax. Substring matching is deliberately conservative: it may refuse a kernel that
/// merely names, say, a variable `plasma` (contains `asm`), never the reverse.
pub const HIP_RESIDUE: &[&str] = &[
    "__shared__",
    "HIP_DYNAMIC_SHARED",
    "atomic",
    "__syncthreads",
    "__threadfence",
    "__shfl",
    "__ballot",
    "__activemask",
    "__lane_id",
    "__builtin_amdgcn",
    "hipblas",
    "rocblas",
    "miopen",
    "hipLaunchKernelGGL",
    "<<<",
    "printf",
    "asm",
];

/// HIP: `__global__` kernels, `out` / `in{K}` buffers (the same naming convention as the shared
/// walkers' default). Math-call tables are the shared defaults (`expf`, `sqrtf`, ...); a
/// fast-math spelling (`__expf`) lifts to nothing, see the tests.
pub const HIP: Frontend = Frontend::new("hip", "__global__", HIP_RESIDUE, "out", "in", parse_hip);
