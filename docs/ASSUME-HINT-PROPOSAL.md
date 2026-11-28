# `Stmt::Assume` as a hint: proposal only (R7-D22; no code written)

Today (0.8.0): `rockfish-hip-emit` omits `Assume` (the Unpopped 0.64.0 CHANGELOG permits ignoring it).
Omission writes nothing into emitted HIP, so it carries no patent question.

## What the public clang docs promise
Source: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-assume, read 2026-11-28 through a
summariser (quotes capped; the section was read verbatim, the page was not read in full).
- Signature `void __builtin_assume(bool);`; "to provide the optimizer with a boolean invariant that is defined to be true."
- "If the condition is violated during execution, the behavior is undefined." This matches Unpopped's
  `Assume` (UB on a device when false; a typed trap in the oracle).
- "The argument itself is never evaluated, so any side effects of the expression will be discarded."
  Unpopped's `cond` is a pure `ScalarExpr`, so nothing is lost.
- Feature test `__has_builtin(__builtin_assume)`.
- NOT on the page as read: any statement about HIP or device code for this builtin, or about which
  optimisations a given AMD target performs. Whether hipcc's clang honours it identically is UNVERIFIED
  and so is any speed effect (no device here).

## What the emitted text would be
One line at the statement's position, using the emitter's existing spelling of `cond`:
`__builtin_assume(<cond>);`, e.g. for `threadIdx.x < 32`: `__builtin_assume(((unsigned int)threadIdx.x) < ((unsigned int)32LL));`
(illustrative; the exact text is whatever `expr` spells). The argument must spell as a C++ bool: a
non-comparison `cond` would need an explicit `!= 0`, to be decided by the spelling code.

## Decision asked of the PM
Whether a hint lowering is wanted at all. Costs: it writes an assume statement into emitted HIP, which
omission does not (patent read of that statement: none made, so not cleared; no attorney). Gain: only a
possible optimisation, UNMEASURED. Default if no answer: keep omission. If wanted: one PR, a test that the
text appears, the existing `assume_ignored.rs` byte-equality test replaced by one for the hint, and a
mutant.

— lane: rockfish
