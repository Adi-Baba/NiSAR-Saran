//! Multiplicity-control primitives.

pub fn benjamini_hochberg(p_values: &[f64], q: f64) -> Vec<bool> {
    let mut order: Vec<usize> = (0..p_values.len()).collect(); order.sort_by(|a,b| p_values[*a].total_cmp(&p_values[*b]));
    let mut cutoff=None; for (rank,idx) in order.iter().enumerate() { let t=q*(rank+1) as f64/p_values.len().max(1) as f64; if p_values[*idx] <= t { cutoff=Some(rank); }}
    let Some(k)=cutoff else { return vec![false;p_values.len()] };
    let threshold=p_values[order[k]]; p_values.iter().map(|p| *p<=threshold).collect()
}

/// Monotone Benjamini-Hochberg adjusted p-values in original row order.
pub fn benjamini_hochberg_adjusted(p_values: &[f64]) -> Vec<f64> {
    if p_values.is_empty() { return Vec::new(); }
    let mut order: Vec<usize> = (0..p_values.len()).collect();
    order.sort_by(|a, b| p_values[*a].total_cmp(&p_values[*b]));
    let m = p_values.len() as f64;
    let mut adjusted = vec![1.0; p_values.len()];
    let mut running: f64 = 1.0;
    for rank0 in (0..order.len()).rev() {
        let index = order[rank0];
        let candidate = (p_values[index].clamp(0.0, 1.0) * m / (rank0 + 1) as f64).min(1.0);
        running = running.min(candidate);
        adjusted[index] = running;
    }
    adjusted
}

/// Deterministic Monte-Carlo paired sign-flip test for the absolute mean.
///
/// The plus-one correction prevents a zero p-value. A small local xorshift RNG
/// keeps the audit reproducible without an external random-number dependency.
pub fn sign_flip_mean_p(values: &[f64], permutations: usize, seed: u64) -> Option<f64> {
    let finite: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.is_empty() || permutations == 0 { return None; }
    let observed = (finite.iter().sum::<f64>() / finite.len() as f64).abs();
    let mut state = seed.max(1);
    let mut exceed = 0usize;
    for _ in 0..permutations {
        let mut sum = 0.0;
        for value in &finite {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            sum += if state & 1 == 0 { *value } else { -*value };
        }
        if (sum / finite.len() as f64).abs() >= observed { exceed += 1; }
    }
    Some((exceed + 1) as f64 / (permutations + 1) as f64)
}

pub fn max_statistic(observed: f64, null_maxima: &[f64]) -> Option<f64> {
    if null_maxima.is_empty() { return None; }
    Some((1 + null_maxima.iter().filter(|x| **x >= observed).count()) as f64 / (null_maxima.len()+1) as f64)
}

/// SplitMix64 step: deterministic `std`-only stream for reproducible null
/// draws. The seed and stream id are recorded in the artifact for replay.
pub fn splitmix_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Deterministic block-flip draw: each of `n_blocks` blocks is flipped
/// with probability one half. Distinct `stream` values give independent
/// replicate streams from the same seed. Returns empty for zero blocks.
pub fn block_flip_draw(n_blocks: usize, seed: u64, stream: u64) -> Vec<bool> {
    let mut state = seed.wrapping_add(stream.wrapping_mul(0x9E3779B97F4A7C15)).max(1);
    (0..n_blocks).map(|_| splitmix_next(&mut state) & 1 == 0).collect()
}

/// One-sided plus-one p-value for a held-out improvement statistic
/// (paper 4 covariate-group null): the fraction of joint within-block
/// permutations of the group that improve at least as much as observed.
/// Directional by design — improvement is the only rejection direction —
/// which distinguishes it from the two-sided [`max_statistic`] scan rule.
/// `None` for non-finite observation or an empty null set.
pub fn group_improvement_p(observed: f64, null_improvements: &[f64]) -> Option<f64> {
    if !observed.is_finite() {
        return None;
    }
    let finite: Vec<f64> = null_improvements.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.is_empty() {
        return None;
    }
    let hits = finite.iter().filter(|x| **x >= observed).count();
    Some((1 + hits) as f64 / (1 + finite.len()) as f64)
}

/// Upper-tail detection threshold for a scan family: an observed maximum
/// strictly above this value rejects at familywise level `tail`
/// (`tail = 0.05` gives the 5% max-statistic rule). At most
/// `floor(tail * B)` of the `B` null maxima lie strictly above it.
/// `None` for an empty null set or a non-finite/invalid tail.
pub fn scan_threshold(null_maxima: &[f64], tail: f64) -> Option<f64> {
    if !(tail > 0.0) || !(tail < 1.0) {
        return None;
    }
    let mut finite: Vec<f64> = null_maxima.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.is_empty() {
        return None;
    }
    finite.sort_by(f64::total_cmp);
    let rank = ((1.0 - tail) * finite.len() as f64).ceil() as usize;
    finite.get(rank.min(finite.len()).saturating_sub(1).max(0)).copied()
}

#[cfg(test)]
mod tests { use super::*;
    #[test] fn bh_rejects_smallest_family_members() { assert_eq!(benjamini_hochberg(&[0.001,0.4,0.002],0.05), vec![true,false,true]); }
    #[test] fn max_statistic_uses_plus_one_correction() { assert_eq!(max_statistic(3.0,&[1.0,2.0,4.0,5.0]),Some(3.0/5.0)); }
    #[test] fn adjusted_values_are_monotone_in_rank() {
        let adjusted = benjamini_hochberg_adjusted(&[0.001, 0.4, 0.002]);
        assert!(adjusted[0] <= adjusted[2] && adjusted[2] <= adjusted[1]);
    }
    #[test] fn sign_flip_detects_a_constant_shift() {
        assert!(sign_flip_mean_p(&[2.0; 20], 999, 7).unwrap() < 0.01);
    }
    #[test] fn block_draws_are_reproducible_and_streamed() {
        let a = block_flip_draw(16, 42, 3);
        assert_eq!(a, block_flip_draw(16, 42, 3));
        assert_ne!(a, block_flip_draw(16, 42, 4));
        assert_eq!(block_flip_draw(0, 42, 3), Vec::<bool>::new());
        // Roughly half the blocks flip on a long draw.
        let flips = block_flip_draw(1000, 1, 0).iter().filter(|x| **x).count();
        assert!((flips as i64 - 500).abs() < 100);
    }
    #[test] fn improvement_p_is_one_sided_plus_one() {
        assert_eq!(group_improvement_p(0.5, &[0.1, 0.2, 0.3]), Some(0.25));
        assert_eq!(group_improvement_p(0.05, &[0.1, 0.2, 0.3]), Some(1.0));
        assert_eq!(group_improvement_p(f64::NAN, &[0.1]), None);
        assert_eq!(group_improvement_p(0.5, &[]), None);
    }
    #[test] fn scan_threshold_holds_the_tail() {
        let null: Vec<f64> = (1..=100).map(|v| v as f64).collect();
        let t = scan_threshold(&null, 0.05).unwrap();
        let above = null.iter().filter(|x| **x > t).count();
        assert!(above <= 5, "too many null maxima above threshold: {above}");
        assert_eq!(scan_threshold(&null, 0.0), None);
        assert_eq!(scan_threshold(&null, 1.0), None);
        assert_eq!(scan_threshold(&[], 0.05), None);
    }
}
