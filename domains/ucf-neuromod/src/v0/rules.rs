use ucf_frames::v1::NeuromodulatorSnapshot;

use super::NeuromodInputs;
use crate::v0::field::clamp01;

/// Deterministic rule map for neuromodulator deltas.
///
/// Each channel starts at 0.5 and is shifted by scaled inputs,
/// with maximum signed contribution bounded to +/-0.4 before clamping to [0, 1].
pub fn compute_delta(i: NeuromodInputs) -> NeuromodulatorSnapshot {
    let surprise = clamp01(i.surprise);
    let reward = clamp01(i.reward);
    let threat = clamp01(i.threat);
    let social = clamp01(i.social);

    let dopamine = clamp01(0.5 + 0.35 * reward - 0.25 * threat);
    let serotonin = clamp01(0.5 - 0.20 * surprise - 0.20 * threat + 0.10 * social);
    let norepinephrine = clamp01(0.5 + 0.20 * surprise + 0.30 * threat);
    let acetylcholine = clamp01(0.5 + 0.35 * surprise);
    let oxytocin = clamp01(0.5 + 0.35 * social - 0.30 * threat);
    let endorphin = clamp01(0.5 + 0.30 * reward - 0.30 * threat);
    let stress = clamp01(0.5 + 0.25 * threat + 0.20 * surprise - 0.10 * reward);

    NeuromodulatorSnapshot {
        dopamine,
        serotonin,
        norepinephrine,
        acetylcholine,
        oxytocin,
        endorphin,
        stress,
    }
}
