use ucf_runtime::errors::RuntimeError;
use ucf_types::error_codes::ErrorCode;

#[test]
fn runtime_error_codes_are_stable() {
    let e = RuntimeError::Compute(ucf_compute::ComputeError::NotImplemented);
    assert_eq!(e.code(), ErrorCode::RuntimeCompute);
    assert_eq!(e.code().as_u16(), 1003);
    assert_eq!(e.code().stable_key(), "runtime.compute");
}
