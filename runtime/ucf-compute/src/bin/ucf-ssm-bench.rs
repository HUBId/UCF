use std::collections::BTreeMap;

use ucf_compute::ssm::run_ssm_kernel_benchmarks;

fn main() {
    let benches = run_ssm_kernel_benchmarks();
    let mut out = Vec::new();
    for case in benches {
        let speedup = if case.opt_ns == 0 {
            0.0
        } else {
            case.ref_ns as f64 / case.opt_ns as f64
        };
        let mut row = BTreeMap::new();
        row.insert("iterations", serde_json::json!(case.iterations));
        row.insert("ref_ns", serde_json::json!(case.ref_ns));
        row.insert("opt_ns", serde_json::json!(case.opt_ns));
        row.insert("speedup", serde_json::json!(speedup));
        out.push(row);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "bench": "ssm_kernels_v1_1",
            "cases": out
        }))
        .expect("json")
    );
}
