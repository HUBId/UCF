use crate::types::{EdgeId, Hash32, PopId, RegionId, Q16};

#[test]
fn id_display_formats_match() {
    assert_eq!(RegionId::new(7).to_string(), "region:7");
    assert_eq!(PopId::new(11).to_string(), "pop:11");
    assert_eq!(EdgeId::new(3).to_string(), "edge:3");
}

#[test]
fn q16_from_i32_and_to_f32_are_sane() {
    let one = Q16::from_i32(1);
    assert!((one.to_f32() - 1.0).abs() < f32::EPSILON);
}

#[test]
fn q16_mul_small_values_is_correct() {
    let a = Q16::from_f32(1.5);
    let b = Q16::from_f32(2.0);
    let result = a.mul(b);
    assert!((result.to_f32() - 3.0).abs() < 0.000_1);
}

#[test]
fn q16_saturating_add_clamps_on_overflow() {
    let near_max = Q16::from_raw(i32::MAX - 1);
    let two = Q16::from_raw(2);
    let out = near_max.saturating_add(two);
    assert_eq!(out.raw(), i32::MAX);
}

#[test]
fn hash32_display_is_64_hex_chars() {
    let sample = Hash32::from_bytes([0xAB; 32]);
    let s = sample.to_string();
    assert_eq!(s.len(), 64);
    assert!(s
        .chars()
        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
}
