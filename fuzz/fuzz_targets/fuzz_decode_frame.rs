#![no_main]

use libfuzzer_sys::fuzz_target;

const MAX_INPUT: usize = 64 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT {
        return;
    }
    let _ = serde_json::from_slice::<ucf_types::v1::spec::ControlFrame>(data);
});
