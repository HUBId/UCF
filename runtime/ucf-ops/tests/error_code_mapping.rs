use ucf_ops::OpsError;
use ucf_types::error_codes::ErrorCode;

#[test]
fn ops_error_codes_are_stable() {
    let e = OpsError::Invalid("x".to_string());
    assert_eq!(e.code(), ErrorCode::OpsInvalid);
    assert_eq!(e.code().as_u16(), 2005);
    assert_eq!(e.code().stable_key(), "ops.invalid");
}
