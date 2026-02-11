#![no_main]

use libfuzzer_sys::fuzz_target;
use ucf_runtime::sandbox::{decode_ipc_envelope, DecodeCaps};

fuzz_target!(|data: &[u8]| {
    let _ = decode_ipc_envelope(data, DecodeCaps::caps_ipc());
});
