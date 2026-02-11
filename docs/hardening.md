# UCF Hardening v0

## DecodeCaps policy

Decoder entrypoints use explicit caps to prevent unbounded allocations and parser DoS:

- `max_bytes`: maximum total input bytes.
- `max_vec_len`: maximum decoded byte/vector length.
- `max_str_len`: maximum string byte length (reject, no truncation).
- `max_map_len`: maximum list/map item count.

Profiles:

- `caps_default()` for normal runtime decode.
- `caps_ipc()` for IPC envelopes.
- `caps_ess()` for tighter ESS-facing decode.

## Property tests

Property tests run deterministically with fixed case count.

```bash
PROPTEST_CASES=256 cargo test --workspace
```

## Fuzzing locally

Install and run corpus-only smoke fuzz (nightly toolchain required by `cargo-fuzz`):

```bash
rustup toolchain install nightly
cargo +nightly install cargo-fuzz --locked
cd fuzz
cargo +nightly fuzz run fuzz_decode_ipc_envelope corpus/fuzz_decode_ipc_envelope -- -runs=32 -seed=1
cargo +nightly fuzz run fuzz_decode_capability_token corpus/fuzz_decode_capability_token -- -runs=32 -seed=1
cargo +nightly fuzz run fuzz_decode_sandbox_call_envelope corpus/fuzz_decode_sandbox_call_envelope -- -runs=32 -seed=1
```

On stable CI we run a compile-only smoke check:

```bash
cargo check --manifest-path fuzz/Cargo.toml --bins
```

## Add a new fuzz target

1. Add a new `[[bin]]` entry in `fuzz/Cargo.toml`.
2. Add `fuzz/fuzz_targets/<target>.rs` and call only safe decode entrypoints with caps.
3. Add a minimal deterministic corpus in `fuzz/corpus/<target>/`.
4. Extend CI smoke commands.

## Reproducing crashes

If fuzz reports a crashing input file, replay with:

```bash
cd fuzz
cargo +nightly fuzz run <target> <path-to-crash-file> -- -runs=1 -seed=1
```

For manual triage, `DecodeError` exposes bounded fields (`kind`, `at`, `context`) without raw secret dumps.
