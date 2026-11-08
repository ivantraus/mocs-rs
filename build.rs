use std::{
    collections::BTreeSet,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

fn output(program: &str, args: &[&str]) -> Option<String> {
    let result = Command::new(program).args(args).output().ok()?;
    if !result.status.success() {
        return None;
    }
    let value = String::from_utf8(result.stdout).ok()?.trim().to_string();
    Some(value)
}

fn main() {
    let root = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    for name in ["src", "scripts", "examples", "docs", "tests"] {
        println!("cargo:rerun-if-changed={}", root.join(name).display());
    }
    for path in git_metadata_paths(&root) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    if let Some(revision) = output("git", &["rev-parse", "--short=12", "HEAD"]) {
        println!("cargo:rustc-env=MOCS_GIT_REVISION={revision}");
    }
    if let Some(rustc) = output("rustc", &["--version", "--verbose"]) {
        println!(
            "cargo:rustc-env=MOCS_RUSTC_VERSION={}",
            rustc.replace('\n', "; ")
        );
    }
    for (name, key) in [
        ("TARGET", "MOCS_BUILD_TARGET"),
        ("PROFILE", "MOCS_BUILD_PROFILE"),
    ] {
        if let Ok(value) = std::env::var(name) {
            println!("cargo:rustc-env={key}={value}");
        }
    }

    let mut files = BTreeSet::new();
    for name in ["src", "scripts", "examples", "docs", "tests"] {
        collect(&root.join(name), &mut files);
    }
    for name in [
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "rust-toolchain.toml",
        "LICENSE",
        "NOTICE",
    ] {
        let path = root.join(name);
        if path.is_file() {
            files.insert(path);
        }
    }
    let mut payload = Vec::new();
    for file in &files {
        println!("cargo:rerun-if-changed={}", file.display());
        let relative = file
            .strip_prefix(&root)
            .expect("source file below crate root")
            .to_string_lossy();
        payload.extend_from_slice(relative.as_bytes());
        payload.push(0);
        if let Ok(bytes) = fs::read(file) {
            payload.extend_from_slice(&bytes);
        }
        payload.push(0);
    }
    if let Some(id) = hash_payload(&payload) {
        println!("cargo:rustc-env=MOCS_SOURCE_TREE_ID={id}");
    }
    let dirty = output("git", &["status", "--porcelain", "--untracked-files=all"])
        .is_some_and(|s| !s.is_empty());
    println!("cargo:rustc-env=MOCS_SOURCE_DIRTY={dirty}");
}

fn git_metadata_paths(root: &Path) -> Vec<PathBuf> {
    let git_path = |args: &[&str]| {
        let result = Command::new("git")
            .current_dir(root)
            .args(args)
            .output()
            .ok()?;
        if !result.status.success() {
            return None;
        }
        let path = PathBuf::from(String::from_utf8(result.stdout).ok()?.trim());
        Some(if path.is_absolute() {
            path
        } else {
            root.join(path)
        })
    };
    let mut paths = Vec::new();
    if let Some(path) = git_path(&["rev-parse", "--git-path", "HEAD"]) {
        paths.push(path);
    }
    if let Some(reference) = output_in(root, "git", &["symbolic-ref", "--quiet", "HEAD"])
        && let Some(path) = git_path(&["rev-parse", "--git-path", reference.trim()])
    {
        paths.push(path);
    }
    if let Some(path) = git_path(&["rev-parse", "--git-path", "index"]) {
        paths.push(path);
    }
    if let Some(path) = git_path(&["rev-parse", "--git-path", "packed-refs"]) {
        paths.push(path);
    }
    paths
}

fn output_in(root: &Path, program: &str, args: &[&str]) -> Option<String> {
    let result = Command::new(program)
        .current_dir(root)
        .args(args)
        .output()
        .ok()?;
    result
        .status
        .success()
        .then(|| {
            String::from_utf8(result.stdout)
                .ok()
                .map(|s| s.trim().to_owned())
        })
        .flatten()
}

fn collect(path: &Path, out: &mut BTreeSet<PathBuf>) {
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name == "__pycache__") {
                continue;
            }
            collect(&path, out);
        } else if path.is_file() && path.extension().is_none_or(|extension| extension != "pyc") {
            out.insert(path);
        }
    }
}

fn hash_payload(payload: &[u8]) -> Option<String> {
    let mut child = Command::new("git")
        .args(["hash-object", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .ok()?;
    child.stdin.take()?.write_all(payload).ok()?;
    let result = child.wait_with_output().ok()?;
    result
        .status
        .success()
        .then(|| String::from_utf8_lossy(&result.stdout).trim().to_owned())
}
