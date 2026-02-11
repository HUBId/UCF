#![no_main]

use libfuzzer_sys::fuzz_target;
use ucf_policy::capability::{decode_capability_token, DecodeCaps};

fuzz_target!(|data: &[u8]| {
    let _ = decode_capability_token(data, DecodeCaps::caps_default());
});
