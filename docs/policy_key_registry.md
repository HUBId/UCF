# Policy Key Registry v1

| key | type | range | default | module |
|---|---|---|---|---|
| governor_tier_1_q | u64 | 0..=10000 | 2500 | governor |
| governor_tier_2_q | u64 | 0..=10000 | 5000 | governor |
| governor_tier_3_q | u64 | 0..=10000 | 7500 | governor |
| ebm_high_risk_q | u64 | 0..=10000 | 7000 | ebm |
| ebm_low_risk_q | u64 | 0..=10000 | 3000 | ebm |
| world_vljepa_min_windows | u16 | 1..=128 | 2 | world_model |
| world_vljepa_drift_alarm_rate_max_q | UQ0_16 | 0..=10000 | 500 | world_model |
| ssm_opt_drift_alarm_rate_max_q | UQ0_16 | 0..=10000 | 500 | ssm |