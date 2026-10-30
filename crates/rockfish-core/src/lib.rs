//! Runtime loading for the Rockfish crates.
//!
//! Nothing here links against ROCm at build time. A [`LibrarySpec`] names a
//! library once, per operating system; [`Library::open`] searches for it and
//! returns a [`LoadError`] instead of panicking when it is absent, and
//! [`Library::symbol`] does the same for a single missing function.
//!
//! The search order is: the `ROCKFISH_<NAME>_PATH` variable (a file or a
//! directory), then `HIP_PATH` and `ROCM_PATH` (their `bin` on Windows or
//! `lib` on Linux, then the root itself), then the platform loader's own
//! search path.

use std::ffi::{OsStr, OsString};
use std::fmt;
use std::path::{Path, PathBuf};

/// The names a library goes by on each supported operating system.
///
/// Names are tried in order, so list the newest first. An operating system
/// with no entry has no ROCm, and loading fails with
/// [`LoadError::UnsupportedPlatform`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LibrarySpec {
    /// Short name used in error messages and in the `ROCKFISH_<NAME>_PATH`
    /// variable, for example `"HIP"`.
    pub logical_name: &'static str,
    /// File names to try on Windows.
    pub windows: &'static [&'static str],
    /// File names to try on Linux.
    pub linux: &'static [&'static str],
}

impl LibrarySpec {
    /// The file names for the operating system this crate was built for.
    pub fn names(&self) -> &'static [&'static str] {
        if cfg!(windows) {
            self.windows
        } else if cfg!(target_os = "linux") {
            self.linux
        } else {
            &[]
        }
    }

    /// The environment variable that overrides the search, for example
    /// `ROCKFISH_HIP_PATH`.
    pub fn path_variable(&self) -> String {
        format!("ROCKFISH_{}_PATH", self.logical_name.to_ascii_uppercase())
    }
}

/// Why a library or one of its symbols could not be used.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LoadError {
    /// This operating system has no names for the library.
    #[error("{library}: no library names are defined for this operating system")]
    UnsupportedPlatform {
        /// The [`LibrarySpec::logical_name`].
        library: &'static str,
    },
    /// Every candidate failed to load.
    #[error("{library}: not found; tried {}", format_tried(.tried))]
    LibraryNotFound {
        /// The [`LibrarySpec::logical_name`].
        library: &'static str,
        /// Each candidate and the loader's reason for rejecting it.
        tried: Vec<(OsString, String)>,
    },
    /// The library loaded but does not export the symbol.
    #[error("{library}: symbol `{symbol}` is missing: {reason}")]
    SymbolMissing {
        /// The [`LibrarySpec::logical_name`].
        library: &'static str,
        /// The symbol that was asked for.
        symbol: String,
        /// The loader's reason.
        reason: String,
    },
}

fn format_tried(tried: &[(OsString, String)]) -> String {
    let mut out = String::new();
    for (index, (candidate, reason)) in tried.iter().enumerate() {
        if index > 0 {
            out.push_str("; ");
        }
        out.push_str(&format!("{} ({reason})", Path::new(candidate).display()));
    }
    if out.is_empty() {
        out.push_str("nothing");
    }
    out
}

/// The ordered list of things to hand to the platform loader.
///
/// `env` looks up an environment variable; it is a parameter so the order can
/// be tested without touching the process environment. `is_file` reports
/// whether a path names an existing file.
pub fn candidates(
    spec: &LibrarySpec,
    env: &dyn Fn(&str) -> Option<OsString>,
    is_file: &dyn Fn(&Path) -> bool,
) -> Vec<OsString> {
    let names = spec.names();
    let mut out: Vec<OsString> = Vec::new();

    if let Some(value) = env(&spec.path_variable()) {
        let path = PathBuf::from(value);
        if is_file(&path) {
            out.push(path.into_os_string());
        } else {
            out.extend(names.iter().map(|name| path.join(name).into_os_string()));
        }
    }

    let sub = if cfg!(windows) { "bin" } else { "lib" };
    for variable in ["HIP_PATH", "ROCM_PATH"] {
        if let Some(root) = env(variable) {
            let root = PathBuf::from(root);
            for dir in [root.join(sub), root] {
                out.extend(names.iter().map(|name| dir.join(name).into_os_string()));
            }
        }
    }

    out.extend(names.iter().map(|name| OsString::from(*name)));
    out
}

/// A loaded library. Symbols borrowed from it stay valid while it is alive.
pub struct Library {
    spec: LibrarySpec,
    inner: libloading::Library,
    loaded_from: OsString,
}

impl fmt::Debug for Library {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Library")
            .field("library", &self.spec.logical_name)
            .field("loaded_from", &self.loaded_from)
            .finish()
    }
}

impl Library {
    /// Search for the library as described in the crate documentation and load
    /// the first candidate that works.
    pub fn open(spec: &LibrarySpec) -> Result<Self, LoadError> {
        let env = |name: &str| std::env::var_os(name);
        let is_file = |path: &Path| path.is_file();
        Self::open_with(spec, &candidates(spec, &env, &is_file))
    }

    /// Load the first working entry of an explicit candidate list.
    pub fn open_with(spec: &LibrarySpec, candidates: &[OsString]) -> Result<Self, LoadError> {
        if spec.names().is_empty() {
            return Err(LoadError::UnsupportedPlatform {
                library: spec.logical_name,
            });
        }
        let mut tried = Vec::new();
        for candidate in candidates {
            // SAFETY: loading a library runs its initialisers. The candidates
            // are the ROCm libraries the caller asked for by name, found via
            // the search order documented on this crate.
            match unsafe { libloading::Library::new(candidate) } {
                Ok(inner) => {
                    return Ok(Self {
                        spec: *spec,
                        inner,
                        loaded_from: candidate.clone(),
                    });
                }
                Err(error) => tried.push((candidate.clone(), error.to_string())),
            }
        }
        Err(LoadError::LibraryNotFound {
            library: spec.logical_name,
            tried,
        })
    }

    /// What the loader was given when it succeeded: a full path or a bare name.
    pub fn loaded_from(&self) -> &OsStr {
        &self.loaded_from
    }

    /// Look up one exported function.
    ///
    /// # Safety
    ///
    /// `T` must be the exact type of the exported symbol, normally an
    /// `unsafe extern "C" fn(..) -> ..` pointer type; a wrong type is
    /// undefined behaviour when called. The returned value must not be used
    /// after this [`Library`] is dropped.
    pub unsafe fn symbol<T: Copy>(&self, name: &str) -> Result<T, LoadError> {
        // SAFETY: forwarded to the caller by this function's contract.
        let found = unsafe { self.inner.get::<T>(name.as_bytes()) };
        found
            .map(|symbol| *symbol)
            .map_err(|error| LoadError::SymbolMissing {
                library: self.spec.logical_name,
                symbol: name.to_owned(),
                reason: error.to_string(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const NEVER: LibrarySpec = LibrarySpec {
        logical_name: "Nothing",
        windows: &["rockfish-no-such-library.dll"],
        linux: &["librockfish-no-such-library.so"],
    };

    const NO_OS: LibrarySpec = LibrarySpec {
        logical_name: "NoOs",
        windows: &[],
        linux: &[],
    };

    // A library every machine of that OS has, so loading has a positive control
    // without building a fake library.
    #[cfg(windows)]
    const SYSTEM: LibrarySpec = LibrarySpec {
        logical_name: "System",
        windows: &["kernel32.dll"],
        linux: &[],
    };
    #[cfg(windows)]
    const SYSTEM_SYMBOL: &str = "GetTickCount";

    #[cfg(target_os = "linux")]
    const SYSTEM: LibrarySpec = LibrarySpec {
        logical_name: "System",
        windows: &[],
        linux: &["libc.so.6"],
    };
    #[cfg(target_os = "linux")]
    const SYSTEM_SYMBOL: &str = "getpid";

    fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<OsString> {
        let map: HashMap<String, OsString> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), OsString::from(v)))
            .collect();
        move |name| map.get(name).cloned()
    }

    #[test]
    fn path_variable_is_upper_cased() {
        assert_eq!(NEVER.path_variable(), "ROCKFISH_NOTHING_PATH");
    }

    #[test]
    fn search_order_is_override_then_roots_then_loader() {
        let spec = LibrarySpec {
            logical_name: "Hip",
            windows: &["a.dll"],
            linux: &["liba.so"],
        };
        let name = spec.names()[0];
        let env = env_of(&[
            ("ROCKFISH_HIP_PATH", "over"),
            ("HIP_PATH", "hip"),
            ("ROCM_PATH", "rocm"),
        ]);
        let got = candidates(&spec, &env, &|_| false);
        let sub = if cfg!(windows) { "bin" } else { "lib" };
        let want: Vec<OsString> = [
            Path::new("over").join(name),
            Path::new("hip").join(sub).join(name),
            Path::new("hip").join(name),
            Path::new("rocm").join(sub).join(name),
            Path::new("rocm").join(name),
            PathBuf::from(name),
        ]
        .into_iter()
        .map(PathBuf::into_os_string)
        .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn override_naming_a_file_is_used_as_is() {
        let env = env_of(&[("ROCKFISH_NOTHING_PATH", "somewhere/exact-file")]);
        let got = candidates(&NEVER, &env, &|_| true);
        assert_eq!(got[0], OsString::from("somewhere/exact-file"));
    }

    #[test]
    fn missing_library_is_an_error_listing_every_candidate() {
        let error = Library::open_with(&NEVER, &candidates(&NEVER, &env_of(&[]), &|_| false))
            .expect_err("no such library");
        match error {
            LoadError::LibraryNotFound { library, tried } => {
                assert_eq!(library, "Nothing");
                assert_eq!(tried.len(), 1);
            }
            other => panic!("wrong error: {other}"),
        }
    }

    #[test]
    fn platform_without_names_is_reported() {
        let error = Library::open_with(&NO_OS, &[]).expect_err("no names");
        assert!(matches!(error, LoadError::UnsupportedPlatform { .. }));
    }

    #[cfg(any(windows, target_os = "linux"))]
    #[test]
    fn loads_a_real_library_and_finds_a_real_symbol() {
        let library = Library::open(&SYSTEM).expect("system library loads");
        // SAFETY: `GetTickCount` and `getpid` both take no arguments and return
        // a 32-bit integer.
        let function: unsafe extern "C" fn() -> u32 =
            unsafe { library.symbol(SYSTEM_SYMBOL) }.expect("symbol is present");
        let _ = function; // resolving is the test; calling is not needed
    }

    #[cfg(any(windows, target_os = "linux"))]
    #[test]
    fn missing_symbol_is_an_error_not_a_panic() {
        let library = Library::open(&SYSTEM).expect("system library loads");
        // SAFETY: the symbol does not exist, so the type is never used.
        let result: Result<unsafe extern "C" fn(), _> =
            unsafe { library.symbol("rockfish_no_such_symbol") };
        match result {
            Err(LoadError::SymbolMissing { symbol, .. }) => {
                assert_eq!(symbol, "rockfish_no_such_symbol")
            }
            other => panic!("wrong result: {:?}", other.map(|_| ())),
        }
    }
}
