use std::fs;
use std::path::Path;

#[test]
fn no_direct_std_fs_usage_outside_wrappers() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut bad = Vec::new();
    for entry in fs::read_dir(&root).expect("read src") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let file = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if matches!(file, "sandbox_fs.rs" | "io_caps.rs") {
            continue;
        }
        let text = fs::read_to_string(&path).expect("read file");
        if text.contains("std::fs") || text.contains("use std::fs") || text.contains("fs::read(") {
            bad.push(file.to_string());
        }
    }
    assert!(bad.is_empty(), "direct std::fs usage in: {bad:?}");
}
