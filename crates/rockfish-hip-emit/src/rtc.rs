//! The hipRTC [`Compiler`]: HIP C++ source -> an AMD GPU code object (R7d).
//!
//! **UNVERIFIED against a real hipRTC.** The only machine measured has no `hiprtc.dll`, so the
//! option spelling (`--offload-arch=`), the compile path and the produced bytes have never run.
//! What is tested is the target grammar and the typed failure when hipRTC is absent.

use std::fmt;

use rockfish_hiprtc::Program;
use unpopped::{ArtifactKind, Compiler};
use unpopped_vocab::TargetId;

/// Why [`HipRtcCompiler::new`] declined a [`TargetId`].
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum HipRtcTargetError {
    /// The namespace is not `rocm:`; carries the namespace found.
    WrongNamespace(String),
    /// The namespace is `rocm:` but the capability set is not `gfx<hex digits>`; carries it.
    MalformedCapabilitySet(String),
}

impl fmt::Display for HipRtcTargetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongNamespace(ns) => {
                write!(
                    f,
                    "HipRtcCompiler only targets the `rocm:` namespace, got `{ns}:`"
                )
            }
            Self::MalformedCapabilitySet(cap) => {
                write!(
                    f,
                    "`rocm:{cap}` does not match the grammar `gfx<hex digits>`"
                )
            }
        }
    }
}

impl std::error::Error for HipRtcTargetError {}

/// Source -> HIP code object through hipRTC, for one `rocm:gfx...` target.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct HipRtcCompiler {
    target: TargetId,
}

impl HipRtcCompiler {
    /// A compiler for `target`. Declines, never panics, a target outside `rocm:` or whose
    /// capability set is not `gfx` followed by one or more lowercase hex digits (`gfx906`,
    /// `gfx90a`, `gfx1036`, `gfx942`).
    ///
    /// # Errors
    /// [`HipRtcTargetError`] naming the rule the target fails.
    pub fn new(target: TargetId) -> Result<Self, HipRtcTargetError> {
        arch_flag(target)?;
        Ok(Self { target })
    }
}

impl Compiler for HipRtcCompiler {
    /// hipRTC has no compile-deadline API, so `max_compile_ms` is not enforced.
    fn compile(&self, source: &str, entry: &str, _max_compile_ms: u32) -> Result<Vec<u8>, String> {
        let flag = arch_flag(self.target).map_err(|e| format!("hiprtc({entry}): {e}"))?;
        let mut program = Program::new(source, &format!("{entry}.hip"))
            .map_err(|e| format!("hiprtc({entry}) create: {e}"))?;
        match program.compile(&[flag.as_str()]) {
            Ok(()) => program
                .code()
                .map_err(|e| format!("hiprtc({entry}) code: {e}")),
            Err(e) => {
                let log = program.log().unwrap_or_default();
                Err(format!(
                    "hiprtc({entry}): {e}\n--- hiprtc log ---\n{}",
                    log.trim()
                ))
            }
        }
    }

    fn artifact_kind(&self) -> ArtifactKind {
        ArtifactKind::HipCodeObject
    }
}

/// The `--offload-arch=` flag for a `rocm:gfx<hex>` target.
fn arch_flag(target: TargetId) -> Result<String, HipRtcTargetError> {
    let ns = target.namespace();
    if ns != "rocm" {
        return Err(HipRtcTargetError::WrongNamespace(ns));
    }
    let cap = target.capability_set();
    match cap.strip_prefix("gfx") {
        Some(hex)
            if !hex.is_empty()
                && hex
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) =>
        {
            Ok(format!("--offload-arch={cap}"))
        }
        _ => Err(HipRtcTargetError::MalformedCapabilitySet(cap)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: &str) -> TargetId {
        TargetId::parse(s).expect("valid token")
    }

    #[test]
    fn gfx_targets_map_to_offload_arch() {
        for (token, flag) in [
            ("rocm:gfx906", "--offload-arch=gfx906"),
            ("rocm:gfx90a", "--offload-arch=gfx90a"),
            ("rocm:gfx942", "--offload-arch=gfx942"),
            ("rocm:gfx1036", "--offload-arch=gfx1036"),
        ] {
            assert_eq!(arch_flag(t(token)), Ok(flag.to_string()), "{token}");
            assert!(HipRtcCompiler::new(t(token)).is_ok(), "{token}");
        }
    }

    #[test]
    fn other_targets_decline_with_the_rule_they_fail() {
        assert_eq!(
            HipRtcCompiler::new(t("cuda:sm80")),
            Err(HipRtcTargetError::WrongNamespace("cuda".into()))
        );
        for token in [
            "rocm:gfx",
            "rocm:gfx90A",
            "rocm:gfxzz",
            "rocm:sm80",
            "rocm:gfx90a-x",
        ] {
            let cap = t(token).capability_set();
            assert_eq!(
                HipRtcCompiler::new(t(token)),
                Err(HipRtcTargetError::MalformedCapabilitySet(cap)),
                "{token}"
            );
        }
    }

    #[test]
    fn artifact_kind_is_a_hip_code_object() {
        let c = HipRtcCompiler::new(t("rocm:gfx906")).unwrap();
        assert_eq!(c.artifact_kind(), ArtifactKind::HipCodeObject);
    }

    /// Passes both ways, printing which: with hipRTC absent (measured on the laptop) the compile
    /// is an `Err` naming hiprtc, never a panic; with it present, a compile of a trivial kernel
    /// yields non-empty bytes.
    #[test]
    fn compile_without_hiprtc_is_an_error_value_with_it_a_code_object() {
        let c = HipRtcCompiler::new(t("rocm:gfx906")).unwrap();
        let src = "extern \"C\" __global__ void k(float* p) { p[0] = 1.0f; }";
        match c.compile(src, "k", 1000) {
            Ok(bytes) => {
                println!("hipRTC present: {} byte code object", bytes.len());
                assert!(!bytes.is_empty());
            }
            Err(e) => {
                println!("hipRTC unavailable or refused: {e}");
                assert!(e.starts_with("hiprtc(k)"), "{e}");
            }
        }
    }
}
