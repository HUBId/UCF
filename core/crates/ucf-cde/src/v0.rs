#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Edge {
    pub src: VarId,
    pub dst: VarId,
}

pub type VarId = u16;

#[derive(Clone, Debug, PartialEq)]
pub struct Hypothesis {
    pub edge: Edge,
    pub score: f32,
    pub conf: f32,
    pub last_update_ms: u64,
    pub seen_obs: u32,
    pub seen_int: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    pub now_ms: u64,
    pub vars: Vec<(VarId, f32)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Intervention {
    pub now_ms: u64,
    pub do_set: Vec<(VarId, f32)>,
    pub measured: Vec<(VarId, f32)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CdeCfg {
    pub promote_conf: f32,
    pub decay_per_s: f32,
    pub learn_rate: f32,
    pub max_edges: usize,
}

impl Default for CdeCfg {
    fn default() -> Self {
        Self {
            promote_conf: 0.75,
            decay_per_s: 0.02,
            learn_rate: 0.10,
            max_edges: 256,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CdeState {
    pub hyps: Vec<Hypothesis>,
    pub last_obs: Option<Observation>,
    pub last_int: Option<Intervention>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CdeUpdateKind {
    None,
    Updated { changed: usize, top_conf: f32 },
    Pruned { pruned: usize },
}

pub fn on_observation(state: &mut CdeState, cfg: CdeCfg, obs: Observation) -> CdeUpdateKind {
    let mut changed = 0usize;
    if let Some(last) = state.last_obs.as_ref() {
        let mut deltas = Vec::new();
        for (var, value) in &obs.vars {
            if let Some((_, prev)) = last.vars.iter().find(|(v, _)| v == var) {
                deltas.push((*var, *value - *prev));
            }
        }

        deltas.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()).then_with(|| a.0.cmp(&b.0)));
        if deltas.len() >= 2 {
            let (var_a, delta_a) = deltas[0];
            let (var_b, delta_b) = deltas[1];
            let (src, dst, dsrc, ddst) = if delta_a.abs() >= delta_b.abs() {
                (var_a, var_b, delta_a, delta_b)
            } else {
                (var_b, var_a, delta_b, delta_a)
            };

            upsert_hyp(state, cfg, obs.now_ms, Edge { src, dst }, dsrc, ddst, true);
            changed += 1;
        }
    }

    state.last_obs = Some(obs);

    if changed > 0 {
        CdeUpdateKind::Updated {
            changed,
            top_conf: top_conf(&state.hyps),
        }
    } else {
        CdeUpdateKind::None
    }
}

pub fn on_intervention(state: &mut CdeState, cfg: CdeCfg, intv: Intervention) -> CdeUpdateKind {
    let mut changed = 0usize;
    for (a, _) in &intv.do_set {
        for (b, _) in &intv.measured {
            upsert_hyp_int(state, cfg, intv.now_ms, Edge { src: *a, dst: *b });
            changed += 1;
        }
    }

    state.last_int = Some(intv);

    if changed > 0 {
        CdeUpdateKind::Updated {
            changed,
            top_conf: top_conf(&state.hyps),
        }
    } else {
        CdeUpdateKind::None
    }
}

pub fn tick_decay(state: &mut CdeState, cfg: CdeCfg, now_ms: u64) -> CdeUpdateKind {
    let mut pruned = 0usize;

    for hyp in &mut state.hyps {
        let dt_ms = now_ms.saturating_sub(hyp.last_update_ms);
        let dt_s = (dt_ms as f32) / 1000.0;
        hyp.conf = (hyp.conf - cfg.decay_per_s * dt_s).clamp(0.0, 1.0);
        hyp.last_update_ms = now_ms;
    }

    let before_prune = state.hyps.len();
    state.hyps.retain(|h| h.conf >= 0.05);
    pruned += before_prune.saturating_sub(state.hyps.len());

    if state.hyps.len() > cfg.max_edges {
        state.hyps.sort_by(|a, b| {
            b.conf
                .total_cmp(&a.conf)
                .then_with(|| a.edge.src.cmp(&b.edge.src))
        });
        let over = state.hyps.len() - cfg.max_edges;
        state.hyps.truncate(cfg.max_edges);
        pruned += over;
    }

    if pruned > 0 {
        CdeUpdateKind::Pruned { pruned }
    } else {
        CdeUpdateKind::None
    }
}

fn upsert_hyp(
    state: &mut CdeState,
    cfg: CdeCfg,
    now_ms: u64,
    edge: Edge,
    delta_src: f32,
    delta_dst: f32,
    is_obs: bool,
) {
    let sign = if delta_src * delta_dst >= 0.0 {
        1.0
    } else {
        -1.0
    };
    if let Some(h) = state.hyps.iter_mut().find(|h| h.edge == edge) {
        h.score = (h.score + sign * cfg.learn_rate).clamp(-1.0, 1.0);
        h.conf = (h.conf + cfg.learn_rate * 0.5).clamp(0.0, 1.0);
        h.last_update_ms = now_ms;
        if is_obs {
            h.seen_obs = h.seen_obs.saturating_add(1);
        }
    } else {
        state.hyps.push(Hypothesis {
            edge,
            score: (sign * cfg.learn_rate).clamp(-1.0, 1.0),
            conf: (cfg.learn_rate * 0.5).clamp(0.0, 1.0),
            last_update_ms: now_ms,
            seen_obs: u32::from(is_obs),
            seen_int: 0,
        });
    }
}

fn upsert_hyp_int(state: &mut CdeState, cfg: CdeCfg, now_ms: u64, edge: Edge) {
    if let Some(h) = state.hyps.iter_mut().find(|h| h.edge == edge) {
        h.score = (h.score + cfg.learn_rate).clamp(-1.0, 1.0);
        h.conf = (h.conf + cfg.learn_rate).clamp(0.0, 1.0);
        h.last_update_ms = now_ms;
        h.seen_int = h.seen_int.saturating_add(1);
    } else {
        state.hyps.push(Hypothesis {
            edge,
            score: cfg.learn_rate.clamp(-1.0, 1.0),
            conf: cfg.learn_rate.clamp(0.0, 1.0),
            last_update_ms: now_ms,
            seen_obs: 0,
            seen_int: 1,
        });
    }
}

fn top_conf(hyps: &[Hypothesis]) -> f32 {
    hyps.iter()
        .map(|h| h.conf)
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_generates_hypothesis_after_two_observations() {
        let mut state = CdeState::default();
        let cfg = CdeCfg::default();

        assert_eq!(
            on_observation(
                &mut state,
                cfg,
                Observation {
                    now_ms: 100,
                    vars: vec![(1, 0.1), (2, 0.2), (3, 0.8)],
                },
            ),
            CdeUpdateKind::None
        );

        let upd = on_observation(
            &mut state,
            cfg,
            Observation {
                now_ms: 200,
                vars: vec![(1, 0.9), (2, 0.7), (3, 0.81)],
            },
        );

        assert!(matches!(upd, CdeUpdateKind::Updated { .. }));
        assert!(!state.hyps.is_empty());
    }

    #[test]
    fn intervention_increases_conf_faster_than_observation() {
        let mut state_obs = CdeState::default();
        let mut state_int = CdeState::default();
        let cfg = CdeCfg::default();

        let _ = on_observation(
            &mut state_obs,
            cfg,
            Observation {
                now_ms: 0,
                vars: vec![(1, 0.1), (2, 0.2)],
            },
        );
        let _ = on_observation(
            &mut state_obs,
            cfg,
            Observation {
                now_ms: 100,
                vars: vec![(1, 0.8), (2, 0.7)],
            },
        );

        let _ = on_intervention(
            &mut state_int,
            cfg,
            Intervention {
                now_ms: 100,
                do_set: vec![(1, 1.0)],
                measured: vec![(2, 0.9)],
            },
        );

        assert!(state_int.hyps[0].conf > state_obs.hyps[0].conf);
    }

    #[test]
    fn decay_prunes_below_threshold() {
        let mut state = CdeState {
            hyps: vec![Hypothesis {
                edge: Edge { src: 1, dst: 2 },
                score: 0.5,
                conf: 0.051,
                last_update_ms: 0,
                seen_obs: 1,
                seen_int: 0,
            }],
            ..CdeState::default()
        };
        let cfg = CdeCfg {
            decay_per_s: 0.1,
            ..CdeCfg::default()
        };

        let upd = tick_decay(&mut state, cfg, 1000);
        assert!(matches!(upd, CdeUpdateKind::Pruned { .. }));
        assert!(state.hyps.is_empty());
    }

    #[test]
    fn max_edges_enforced() {
        let mut state = CdeState::default();
        let cfg = CdeCfg {
            max_edges: 2,
            ..CdeCfg::default()
        };

        for i in 0..4u16 {
            let _ = on_intervention(
                &mut state,
                cfg,
                Intervention {
                    now_ms: 10 + u64::from(i),
                    do_set: vec![(i + 1, 1.0)],
                    measured: vec![(100 + i, 0.7)],
                },
            );
        }

        let upd = tick_decay(&mut state, cfg, 1000);
        assert!(matches!(upd, CdeUpdateKind::Pruned { .. }));
        assert!(state.hyps.len() <= 2);
    }
}
