// SPDX-License-Identifier: MIT OR Apache-2.0
//! **No tracked source, test, doc or manifest file contains a NUL byte.**
//!
//! A NUL in a text file makes git and every diff tool call it binary, and in a Rust string literal it
//! is a raw control byte standing where the author meant the two-character escape for zero. It
//! compiles fine, which is why nothing else catches it (Unpopped shipped one in 0.37.1).
//!
//! The scan enumerates `git ls-files` (never a disk walk, which would see other lanes' worktrees) and
//! flags any 0 byte in a file with a text extension. Controls: the matcher must fire on a buffer
//! holding a NUL, and the scan must find a real population of files.

use std::path::{Path, PathBuf};
use std::process::Command;

const EXTENSIONS: &[&str] = &[
    "rs", "md", "toml", "yml", "yaml", "json", "txt", "h", "c", "cpp",
];

fn has_nul(bytes: &[u8]) -> bool {
    bytes.contains(&0u8)
}

/// Tracked files with a text extension, as `(root, relative path)`.
fn tracked_text_files() -> (PathBuf, Vec<PathBuf>) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("the workspace root")
        .to_path_buf();
    let out = Command::new("git")
        .arg("-C")
        .arg(&root)
        .args(["ls-files", "-z"])
        .output()
        .expect("run git ls-files");
    assert!(
        out.status.success(),
        "git ls-files failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let files = out
        .stdout
        .split(|b| *b == 0)
        .filter(|s| !s.is_empty())
        .map(|s| PathBuf::from(String::from_utf8_lossy(s).into_owned()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| EXTENSIONS.contains(&e))
        })
        .collect();
    (root, files)
}

#[test]
fn the_matcher_fires_on_a_nul() {
    assert!(has_nul(&[b'a', 0u8, b'b']), "the control: a NUL is seen");
    assert!(!has_nul(b"plain text"));
}

#[test]
fn no_tracked_text_file_contains_a_nul_byte() {
    let (root, files) = tracked_text_files();
    // The population control: a moved root or an empty index would otherwise scan nothing.
    assert!(
        files.len() > 50,
        "found only {} tracked text files; the scan is not looking at the repository",
        files.len()
    );
    let offenders: Vec<String> = files
        .iter()
        .filter(|f| std::fs::read(root.join(f)).is_ok_and(|b| has_nul(&b)))
        .map(|f| f.display().to_string())
        .collect();
    assert!(
        offenders.is_empty(),
        "NUL bytes in text files (a mistyped backslash-zero escape?): {offenders:#?}"
    );
}
