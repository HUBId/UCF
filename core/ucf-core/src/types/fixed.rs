#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct Q16(i32);

impl Q16 {
    pub const fn from_i32(i: i32) -> Q16 {
        Q16(i.wrapping_shl(16))
    }

    pub fn from_f32(x: f32) -> Q16 {
        if x.is_nan() {
            return Q16(0);
        }

        let scaled = (x as f64) * 65536.0_f64;
        let clamped = scaled.clamp(i32::MIN as f64, i32::MAX as f64);

        let rounded = if clamped >= 0.0 {
            (clamped + 0.5).floor()
        } else {
            (clamped - 0.5).ceil()
        };

        Q16(rounded as i32)
    }

    pub const fn from_raw(raw: i32) -> Q16 {
        Q16(raw)
    }

    pub fn to_f32(self) -> f32 {
        self.0 as f32 / 65536.0_f32
    }

    pub const fn raw(self) -> i32 {
        self.0
    }

    pub const fn saturating_add(self, other: Q16) -> Q16 {
        Q16(self.0.saturating_add(other.0))
    }

    pub const fn saturating_sub(self, other: Q16) -> Q16 {
        Q16(self.0.saturating_sub(other.0))
    }

    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Q16) -> Q16 {
        let lhs = self.0 as i64;
        let rhs = other.0 as i64;
        let product = (lhs * rhs) >> 16;

        if product > i32::MAX as i64 {
            Q16(i32::MAX)
        } else if product < i32::MIN as i64 {
            Q16(i32::MIN)
        } else {
            Q16(product as i32)
        }
    }
}
