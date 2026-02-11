#![no_main]

use libfuzzer_sys::fuzz_target;
use ucf_runtime::sandbox::{decode_sandbox_reply, DecodeCaps};

fuzz_target!(|data: &[u8]| {
    let _ = decode_sandbox_reply(data, DecodeCaps::caps_default());
});
