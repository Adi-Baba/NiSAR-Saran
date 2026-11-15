//! Shared finite-value, quantile, and paired-change primitives.
//!
//! Group A owns this crate (papers 1–2). All statistics are deterministic,
//! dependency-free (`std` only), and treat non-finite values (`NaN`,
//! `±inf`) as missing: they are counted, never silently dropped from the
//! ledger, and never enter a location/scale summary.

/// Gaussian-consistency factor: `MAD * 1.4826` estimates `sigma` for
/// Gaussian data. The raw MAD is always reported alongside the scaled one
/// so no Gaussian assumption is smuggled into the primary result.
pub const MAD_TO_SIGMA: f64 = 1.4826;

/// Floor for the scale denominator of the robust effect. Prevents a
/// degenerate zero-MAD field from producing an infinite claim from
/// numerical dust while keeping the exact-zero case at exactly zero.
pub const EFFECT_EPSILON: f64 = 1e-12;

/// Fixed symmetric trim for the trimmed-mean companion summary.
/// Declared here so it cannot be tuned after ranking.
pub const TRIM_FRACTION: f64 = 0.10;

#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    pub n: usize,
    pub finite: usize,
    pub median: Option<f64>,
    pub mad: Option<f64>,
    pub q05: Option<f64>,
    pub q25: Option<f64>,
    pub q75: Option<f64>,
    pub q95: Option<f64>,
    pub positive_fraction: Option<f64>,
    pub robust_effect: Option<f64>,
}

/// Paired before/after counts plus robust summaries of the difference field.
///
/// `median_delta` / `mad_delta` are in data units; `mad_scaled_delta` is
/// `1.4826 * MAD` (Gaussian-consistent scale); `robust_effect` is the
/// standardized median (see [`robust_effect`]). Trimmed mean and tail
/// quantiles are companions: no single summary is the result on its own.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedSummary {
    pub n_pairs: usize,
    pub finite_before: usize,
    pub finite_after: usize,
    pub finite_pairs: usize,
    pub median_delta: Option<f64>,
    pub mad_delta: Option<f64>,
    pub mad_scaled_delta: Option<f64>,
    pub q05_delta: Option<f64>,
    pub q25_delta: Option<f64>,
    pub q75_delta: Option<f64>,
    pub q95_delta: Option<f64>,
    pub positive_fraction: Option<f64>,
    pub trimmed_mean_delta: Option<f64>,
    pub robust_effect: Option<f64>,
}

pub fn finite(values: &[f64]) -> Vec<f64> {
    values.iter().copied().filter(|x| x.is_finite()).collect()
}

/// Type-7 (linear-interpolation) quantile on all values as given.
///
/// Returns `None` for empty input or `p` outside `[0, 1]`. Non-finite
/// entries are **not** filtered here: filter with [`finite`] first when
/// they represent missing data, or use [`summary`]/[`paired_summary`]
/// which do that internally.
pub fn quantile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() || !(0.0..=1.0).contains(&p) {
        return None;
    }
    quantile_on_sorted(&{
        let mut v = values.to_vec();
        v.sort_by(f64::total_cmp);
        v
    }, p)
}

/// Type-7 quantile of an already-ascending slice. Same contract as
/// [`quantile`] without re-sorting; used by multi-quantile sweeps.
pub fn quantile_on_sorted(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() || !(0.0..=1.0).contains(&p) {
        return None;
    }
    let x = p * (sorted.len() - 1) as f64;
    let lo = x.floor() as usize;
    let hi = x.ceil() as usize;
    Some(if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (sorted[hi] - sorted[lo]) * (x - lo as f64)
    })
}

pub fn median(values: &[f64]) -> Option<f64> {
    quantile(values, 0.5)
}

pub fn mad(values: &[f64]) -> Option<f64> {
    let m = median(values)?;
    let d: Vec<f64> = values.iter().map(|x| (x - m).abs()).collect();
    median(&d)
}

/// Gaussian-consistent scale `1.4826 * MAD`. `None` propagates from
/// [`mad`]; a zero MAD maps to zero (not `None`) so the degenerate case
/// stays distinguishable from missing data.
pub fn mad_scaled(values: &[f64]) -> Option<f64> {
    mad(values).map(|m| MAD_TO_SIGMA * m)
}

/// Fraction of finite values strictly greater than zero. `None` for
/// empty input. Zeros count against the fraction (they are not positive),
/// so a mask-transition field of all zeros reports `Some(0.0)`, not `None`.
pub fn sign_fraction(values: &[f64]) -> Option<f64> {
    let v = finite(values);
    (!v.is_empty()).then(|| v.iter().filter(|x| **x > 0.0).count() as f64 / v.len() as f64)
}

/// Symmetric trimmed mean of the finite values.
///
/// Removes fraction `trim` from **each** tail (`0 <= trim < 0.5`) and
/// averages the rest. `trim = 0` is the mean; large `trim` approaches the
/// median. Returns `None` for empty input, invalid `trim`, or when
/// trimming would remove everything.
pub fn trimmed_mean(values: &[f64], trim: f64) -> Option<f64> {
    if !(0.0..0.5).contains(&trim) {
        return None;
    }
    let mut v = finite(values);
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let k = (trim * v.len() as f64).floor() as usize;
    let kept = &v[k..v.len() - k];
    if kept.is_empty() {
        return None;
    }
    Some(kept.iter().sum::<f64>() / kept.len() as f64)
}

/// Standardized robust effect `median / max(1.4826*MAD, EPSILON)`.
///
/// * both location and scale present, scale above floor → ratio;
/// * scale at/below floor and median exactly zero → `0.0` (no change, no
///   matter how tight the field);
/// * scale at/below floor and median nonzero → signed infinity (a
///   degenerate, perfectly constant nonzero shift — reported, and the
///   accompanying MAD of ~0 keeps it honest);
/// * anything missing → `None`.
pub fn robust_effect(values: &[f64]) -> Option<f64> {
    let m = median(&finite(values))?;
    let s = mad_scaled(values)?;
    if s > EFFECT_EPSILON {
        Some(m / s)
    } else if m == 0.0 {
        Some(0.0)
    } else {
        Some(f64::INFINITY.copysign(m))
    }
}

pub fn summary(values: &[f64]) -> Summary {
    let v = finite(values);
    let m = median(&v);
    let scale = mad(&v);
    Summary {
        n: values.len(),
        finite: v.len(),
        median: m,
        mad: scale,
        q05: quantile(&v, 0.05),
        q25: quantile(&v, 0.25),
        q75: quantile(&v, 0.75),
        q95: quantile(&v, 0.95),
        positive_fraction: (!v.is_empty()).then(|| {
            v.iter().filter(|x| **x > 0.0).count() as f64 / v.len() as f64
        }),
        robust_effect: match (m, scale) {
            (Some(a), Some(b)) if MAD_TO_SIGMA * b > EFFECT_EPSILON => {
                Some(a / (MAD_TO_SIGMA * b))
            }
            (Some(a), Some(_)) => Some(if a == 0.0 {
                0.0
            } else {
                f64::INFINITY.copysign(a)
            }),
            _ => None,
        },
    }
}

pub fn paired_delta(before: &[f64], after: &[f64]) -> Vec<f64> {
    assert_eq!(before.len(), after.len());
    before
        .iter()
        .zip(after)
        .filter_map(|(b, a)| (b.is_finite() && a.is_finite()).then_some(a - b))
        .collect()
}

/// Full paired summary: validity counts plus every robust companion for
/// the difference field. Panics on unpaired lengths (a contract bug, not
/// data): callers must align tiles before calling.
pub fn paired_summary(before: &[f64], after: &[f64]) -> PairedSummary {
    assert_eq!(
        before.len(),
        after.len(),
        "paired arrays must have equal length"
    );
    let finite_before = finite(before).len();
    let finite_after = finite(after).len();
    let deltas = paired_delta(before, after);
    let med = median(&deltas);
    let scale = mad(&deltas);
    PairedSummary {
        n_pairs: before.len(),
        finite_before,
        finite_after,
        finite_pairs: deltas.len(),
        median_delta: med,
        mad_delta: scale,
        mad_scaled_delta: scale.map(|s| MAD_TO_SIGMA * s),
        q05_delta: quantile(&deltas, 0.05),
        q25_delta: quantile(&deltas, 0.25),
        q75_delta: quantile(&deltas, 0.75),
        q95_delta: quantile(&deltas, 0.95),
        positive_fraction: sign_fraction(&deltas),
        trimmed_mean_delta: trimmed_mean(&deltas, TRIM_FRACTION),
        robust_effect: match (med, scale) {
            (Some(a), Some(b)) if MAD_TO_SIGMA * b > EFFECT_EPSILON => {
                Some(a / (MAD_TO_SIGMA * b))
            }
            (Some(a), Some(_)) => Some(if a == 0.0 {
                0.0
            } else {
                f64::INFINITY.copysign(a)
            }),
            _ => None,
        },
    }
}

/// Deterministic sign-flip mean of paired differences.
///
/// `flips[i] == true` negates `deltas[i]`. The permutation source stays
/// outside the statistic so null draws are auditable; pair with
/// [`deterministic_flips`] for reproducible replicates. `None` on length
/// mismatch or empty input.
pub fn sign_flip_mean(deltas: &[f64], flips: &[bool]) -> Option<f64> {
    if deltas.len() != flips.len() || deltas.is_empty() {
        return None;
    }
    let total: f64 = deltas
        .iter()
        .zip(flips)
        .map(|(x, flip)| if *flip { -*x } else { *x })
        .sum();
    Some(total / deltas.len() as f64)
}

/// Monte Carlo permutation p-value `(1 + #{T_b >= T_obs}) / (1 + B)`.
///
/// The `+1` counts the observed statistic as one replicate: the p-value
/// is exact (never zero) under an exchangeable null. Non-finite null
/// replicates are ignored; `None` when the observation is non-finite or
/// no finite null replicate exists.
pub fn permutation_pvalue(observed: f64, null_stats: &[f64]) -> Option<f64> {
    if !observed.is_finite() {
        return None;
    }
    let finite_null: Vec<f64> = null_stats.iter().copied().filter(|x| x.is_finite()).collect();
    if finite_null.is_empty() {
        return None;
    }
    let hits = finite_null.iter().filter(|x| **x >= observed).count();
    Some((1 + hits) as f64 / (1 + finite_null.len()) as f64)
}

/// SplitMix64 step: deterministic, `std`-only PRNG for reproducible null
/// draws. `state` is updated in place and the output word is returned.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `n` reproducible pseudo-random sign bits from `seed`. Top bit of each
/// SplitMix64 word; the seed is recorded in the artifact for replay.
pub fn deterministic_flips(n: usize, seed: u64) -> Vec<bool> {
    let mut state = seed;
    (0..n).map(|_| splitmix64(&mut state) >> 63 == 1).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn summary_ignores_nonfinite() {
        let s = summary(&[1.0, 2.0, f64::NAN, f64::INFINITY]);
        assert_eq!(s.finite, 2);
        assert_eq!(s.median, Some(1.5));
    }
    #[test]
    fn paired_delta_preserves_pairing() {
        assert_eq!(
            paired_delta(&[1.0, f64::NAN, 4.0], &[3.0, 9.0, 1.0]),
            vec![2.0, -3.0]
        );
    }
    #[test]
    fn quantiles_are_interpolated_and_order_independent() {
        assert_eq!(quantile(&[4.0, 1.0, 3.0, 2.0], 0.5), Some(2.5));
        assert_eq!(quantile(&[4.0, 1.0, 3.0, 2.0], 0.0), Some(1.0));
        assert_eq!(quantile(&[4.0, 1.0, 3.0, 2.0], 1.0), Some(4.0));
        assert_eq!(quantile(&[], 0.5), None);
        assert_eq!(quantile(&[1.0], 1.5), None);
        assert_eq!(quantile(&[1.0], f64::NAN), None);
    }
    #[test]
    fn quantile_on_sorted_matches_quantile() {
        let mut v = vec![3.0, 1.0, 2.0, 4.0];
        v.sort_by(f64::total_cmp);
        for p in [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0] {
            assert_eq!(quantile_on_sorted(&v, p), quantile(&v, p));
        }
    }
    #[test]
    fn mad_scales_to_sigma_for_gaussian_shape() {
        // For {-2,-1,0,1,2}: median 0, MAD 1, scaled 1.4826.
        assert_eq!(mad(&[-2.0, -1.0, 0.0, 1.0, 2.0]), Some(1.0));
        assert_eq!(mad_scaled(&[-2.0, -1.0, 0.0, 1.0, 2.0]), Some(1.4826));
    }
    #[test]
    fn robust_effect_uses_scaled_mad_with_floor() {
        // median 2, MAD 1 -> 2/1.4826.
        let e = robust_effect(&[0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!((e - 2.0 / 1.4826).abs() < 1e-12, "got {e}");
        assert_eq!(robust_effect(&[0.0, 0.0, 0.0]), Some(0.0));
        assert_eq!(robust_effect(&[2.0, 2.0, 2.0]), Some(f64::INFINITY));
        assert_eq!(robust_effect(&[-2.0, -2.0, -2.0]), Some(f64::NEG_INFINITY));
        assert_eq!(robust_effect(&[]), None);
        assert_eq!(robust_effect(&[f64::NAN]), None);
    }
    #[test]
    fn trimmed_mean_trims_each_tail() {
        // trim 0.25 of 4 values keeps middle two: (2+3)/2.
        assert_eq!(trimmed_mean(&[1.0, 2.0, 3.0, 100.0], 0.25), Some(2.5));
        assert_eq!(trimmed_mean(&[1.0, 2.0, 3.0], 0.0), Some(2.0));
        assert_eq!(trimmed_mean(&[], 0.1), None);
        assert_eq!(trimmed_mean(&[1.0], 0.5), None);
        assert_eq!(trimmed_mean(&[1.0], -0.1), None);
        // Non-finite values are missing, not zeros.
        assert_eq!(trimmed_mean(&[1.0, f64::NAN, 3.0], 0.0), Some(2.0));
    }
    #[test]
    fn sign_fraction_counts_strictly_positive() {
        assert_eq!(sign_fraction(&[1.0, -1.0, 0.0, 2.0]), Some(0.5));
        assert_eq!(sign_fraction(&[0.0, 0.0]), Some(0.0));
        assert_eq!(sign_fraction(&[]), None);
        assert_eq!(sign_fraction(&[f64::NAN]), None);
    }
    #[test]
    fn paired_summary_keeps_missingness_separate() {
        let s = paired_summary(&[1.0, f64::NAN, 3.0], &[2.0, 4.0, f64::INFINITY]);
        assert_eq!(s.n_pairs, 3);
        assert_eq!(s.finite_before, 2);
        assert_eq!(s.finite_after, 2);
        assert_eq!(s.finite_pairs, 1);
        assert_eq!(s.median_delta, Some(1.0));
        assert_eq!(s.trimmed_mean_delta, Some(1.0));
    }
    #[test]
    fn paired_summary_effect_matches_scaled_rule() {
        // deltas {1,1,3,3,3}: median 3? sorted [1,1,3,3,3] median 3,
        // MAD = median(|d-3|) = median([2,2,0,0,0]) = 0 -> +inf.
        let s = paired_summary(&[0.0; 5], &[1.0, 1.0, 3.0, 3.0, 3.0]);
        assert_eq!(s.median_delta, Some(3.0));
        assert_eq!(s.mad_delta, Some(0.0));
        assert_eq!(s.robust_effect, Some(f64::INFINITY));
    }
    #[test]
    fn sign_flip_is_reproducible_from_supplied_bits() {
        assert_eq!(
            sign_flip_mean(&[1.0, 3.0, -2.0], &[false, true, false]),
            Some(-4.0 / 3.0)
        );
        assert_eq!(sign_flip_mean(&[], &[]), None);
        assert_eq!(sign_flip_mean(&[1.0], &[true, false]), None);
    }
    #[test]
    fn permutation_pvalue_counts_observed_replicate() {
        // obs 2.0, null {1,2,3}: hits {2,3} -> (1+2)/(1+3) = 0.75.
        assert_eq!(permutation_pvalue(2.0, &[1.0, 2.0, 3.0]), Some(0.75));
        // Nothing exceeds -> (1+0)/(1+B), never zero.
        assert_eq!(permutation_pvalue(9.0, &[1.0, 2.0]), Some(1.0 / 3.0));
        assert_eq!(permutation_pvalue(f64::NAN, &[1.0]), None);
        assert_eq!(permutation_pvalue(1.0, &[]), None);
        // Non-finite null draws are ignored, not counted.
        assert_eq!(permutation_pvalue(1.0, &[f64::NAN, 0.0]), Some(1.0 / 2.0));
    }
    #[test]
    fn deterministic_flips_are_stable_and_seed_sensitive() {
        assert_eq!(deterministic_flips(16, 42), deterministic_flips(16, 42));
        assert_ne!(deterministic_flips(64, 1), deterministic_flips(64, 2));
        assert_eq!(deterministic_flips(0, 7), Vec::<bool>::new());
    }
}
