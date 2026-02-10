#![forbid(unsafe_code)]

use std::env;
use ucf_demo::run_cycles;

fn main() {
    let mut cycles = 12u64;
    let mut seed = 42u64;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--cycles" => {
                cycles = args
                    .next()
                    .expect("--cycles expects a value")
                    .parse()
                    .expect("--cycles value must be an integer");
            }
            "--seed" => {
                seed = args
                    .next()
                    .expect("--seed expects a value")
                    .parse()
                    .expect("--seed value must be an integer");
            }
            _ => panic!("unknown arg: {arg}"),
        }
    }

    let summaries = run_cycles(cycles, seed);
    for s in summaries {
        let targets = s
            .delta_targets
            .iter()
            .map(|t| if *t > 0 { '1' } else { '0' })
            .collect::<String>();
        let violations = s
            .violations
            .iter()
            .map(|v| format!("{}:{}:{}", v.rule_id, v.severity, v.reason))
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "cycle={} gamma_bucket={} plv={} lock_window={} surprise={} novelty={} salience={} attention_gain={} learn_rate={} mode={} delta_mass={} targets={} nsr_verdict={} nsr_hits={:?} violations={}",
            s.cycle_id,
            s.gamma_bucket,
            s.plv,
            s.lock_window,
            s.surprise,
            s.novelty,
            s.salience,
            s.attention_gain,
            s.learn_rate,
            s.learn_mode,
            s.delta_mass,
            targets,
            s.nsr_verdict.unwrap_or_default(),
            s.nsr_hit_counts,
            if violations.is_empty() { "none" } else { &violations }
        );
    }
}
