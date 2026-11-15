//! Distribution-shift statistics on deterministic spatial-block samples.
//!
//! Group A owns this crate (paper 2). Scalar summaries: quantile
//! displacement, energy distance, Gaussian MMD (unbiased U-statistic),
//! and 1-Wasserstein distance. Every statistic pairs with
//! [`permutation_pvalue`] under the block-label-swap null; bandwidth
//! selection is frozen in [`pooled_bandwidth`] before any ranking.

use audit_core::{finite, MAD_TO_SIGMA};

/// Floor for the kernel bandwidth in data units. Keeps a degenerate
/// (near-constant) pooled field from collapsing the Gaussian kernel to a
/// spike, which would turn numerical dust into a false distributional
/// difference.
pub const BANDWIDTH_EPSILON: f64 = 1e-9;

pub use audit_core::permutation_pvalue;

pub fn energy_distance(x: &[f64], y: &[f64]) -> Option<f64> {
    let x = finite(x);
    let y = finite(y);
    if x.is_empty() || y.is_empty() {
        return None;
    }
    let cross = x
        .iter()
        .flat_map(|a| y.iter().map(move |b| (a - b).abs()))
        .sum::<f64>()
        / (x.len() * y.len()) as f64;
    let xx = x
        .iter()
        .flat_map(|a| x.iter().map(move |b| (a - b).abs()))
        .sum::<f64>()
        / (x.len() * x.len()) as f64;
    let yy = y
        .iter()
        .flat_map(|a| y.iter().map(move |b| (a - b).abs()))
        .sum::<f64>()
        / (y.len() * y.len()) as f64;
    Some(2.0 * cross - xx - yy)
}

pub fn gaussian_mmd_squared(x: &[f64], y: &[f64], bandwidth: f64) -> Option<f64> {
    let x = finite(x);
    let y = finite(y);
    if x.len() < 2 || y.len() < 2 || !(bandwidth.is_finite() && bandwidth > 0.0) {
        return None;
    }
    let k = |a: f64, b: f64| (-(a - b).powi(2) / (2.0 * bandwidth.powi(2))).exp();
    let xx = x
        .iter()
        .enumerate()
        .flat_map(|(i, a)| {
            x.iter()
                .enumerate()
                .filter_map(move |(j, b)| (i != j).then_some(k(*a, *b)))
        })
        .sum::<f64>()
        / (x.len() * (x.len() - 1)) as f64;
    let yy = y
        .iter()
        .enumerate()
        .flat_map(|(i, a)| {
            y.iter()
                .enumerate()
                .filter_map(move |(j, b)| (i != j).then_some(k(*a, *b)))
        })
        .sum::<f64>()
        / (y.len() * (y.len() - 1)) as f64;
    let xy = x
        .iter()
        .flat_map(|a| y.iter().map(move |b| k(*a, *b)))
        .sum::<f64>()
        / (x.len() * y.len()) as f64;
    Some(xx + yy - 2.0 * xy)
}

/// Quantile displacement `Q_Y(t) − Q_X(t)` on a frozen probability grid.
///
/// Type-7 interpolation via [`audit_core::quantile`]. `None` for an empty
/// grid, any `t` outside `[0, 1]`, or no finite value on either side.
/// Upper-tail entries carry tail-contamination signals that location
/// summaries miss.
pub fn quantile_shift(x: &[f64], y: &[f64], probs: &[f64]) -> Option<Vec<f64>> {
    if probs.is_empty() || probs.iter().any(|p| !(0.0..=1.0).contains(p)) {
        return None;
    }
    let xf = finite(x);
    let yf = finite(y);
    if xf.is_empty() || yf.is_empty() {
        return None;
    }
    probs
        .iter()
        .map(|p| {
            match (
                audit_core::quantile(&xf, *p),
                audit_core::quantile(&yf, *p),
            ) {
                (Some(a), Some(b)) => Some(b - a),
                _ => None,
            }
        })
        .collect()
}

/// 1-Wasserstein distance `∫₀¹ |Q_Y(t) − Q_X(t)| dt`.
///
/// Equal sample sizes use the mean absolute gap of paired order
/// statistics (exact for empirical measures); unequal sizes integrate
/// over `max(nx, ny)` evenly spaced quantile midpoints with type-7
/// interpolation. `None` when either side has no finite value.
pub fn wasserstein1(x: &[f64], y: &[f64]) -> Option<f64> {
    let mut xs = finite(x);
    let mut ys = finite(y);
    if xs.is_empty() || ys.is_empty() {
        return None;
    }
    xs.sort_by(f64::total_cmp);
    ys.sort_by(f64::total_cmp);
    if xs.len() == ys.len() {
        let n = xs.len();
        return Some(
            xs.iter()
                .zip(&ys)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / n as f64,
        );
    }
    let m = xs.len().max(ys.len());
    let mut total = 0.0;
    for i in 0..m {
        let t = (i as f64 + 0.5) / m as f64;
        let qx = audit_core::quantile_on_sorted(&xs, t)?;
        let qy = audit_core::quantile_on_sorted(&ys, t)?;
        total += (qy - qx).abs();
    }
    Some(total / m as f64)
}

/// Frozen bandwidth rule: `max(1.4826 * MAD(pooled), epsilon)`.
///
/// The pooled MAD is computed over the concatenated finite samples; the
/// Gaussian-consistency factor keeps the kernel width on the data's own
/// scale and the epsilon floor keeps degenerate fields reproducible.
/// `None` when the pooled sample is empty. Declared before ranking and
/// stored in the artifact — never re-tuned against candidates.
pub fn pooled_bandwidth(x: &[f64], y: &[f64]) -> Option<f64> {
    let mut pooled = finite(x);
    pooled.extend(finite(y));
    if pooled.is_empty() {
        return None;
    }
    let scale = audit_core::mad(&pooled).map(|m| MAD_TO_SIGMA * m)?;
    Some(scale.max(BANDWIDTH_EPSILON))
}

/// One block-label-swap null replicate for paired samples.
///
/// `blocks[i]` is the spatial block of pair `i`; blocks flagged in
/// `flip` exchange their `x`/`y` labels, the rest stay. Under the paired
/// null (date labels exchangeable within blocks) the recomputed
/// statistic joins the null distribution for [`permutation_pvalue`].
/// `None` on length mismatch or empty input.
pub fn block_label_swap(
    x: &[f64],
    y: &[f64],
    blocks: &[usize],
    flip: &[bool],
) -> Option<(Vec<f64>, Vec<f64>)> {
    if x.len() != y.len() || x.len() != blocks.len() || x.is_empty() {
        return None;
    }
    let n_blocks = blocks.iter().max().map(|m| m + 1).unwrap_or(0);
    if flip.len() != n_blocks {
        return None;
    }
    let mut xs = Vec::with_capacity(x.len());
    let mut ys = Vec::with_capacity(y.len());
    for i in 0..x.len() {
        if flip[blocks[i]] {
            xs.push(y[i]);
            ys.push(x[i]);
        } else {
            xs.push(x[i]);
            ys.push(y[i]);
        }
    }
    Some((xs, ys))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identical_samples_have_zero_energy_distance() {
        assert_eq!(energy_distance(&[1.0, 2.0], &[1.0, 2.0]), Some(0.0));
    }
    #[test]
    fn energy_distance_rejects_empty_sides() {
        assert_eq!(energy_distance(&[], &[1.0]), None);
        assert_eq!(energy_distance(&[f64::NAN], &[1.0]), None);
    }
    #[test]
    fn shifted_samples_have_positive_mmd() {
        assert!(gaussian_mmd_squared(&[0.0, 0.1], &[2.0, 2.1], 1.0).unwrap() > 0.0);
    }
    #[test]
    fn mmd_rejects_degenerate_inputs() {
        assert_eq!(gaussian_mmd_squared(&[0.0], &[2.0, 2.1], 1.0), None);
        assert_eq!(gaussian_mmd_squared(&[0.0, 0.1], &[2.0, 2.1], 0.0), None);
        assert_eq!(gaussian_mmd_squared(&[0.0, 0.1], &[2.0, 2.1], f64::NAN), None);
    }
    #[test]
    fn mmd_diagonal_effect_for_identical_arrays() {
        // The unbiased estimator pairs U-statistic diagonal-free within
        // terms with a V-statistic (diagonal included) cross term. For
        // literally identical arrays x == y the estimate is therefore
        // 2*(off - 1)/n <= 0, where `off` is the mean off-diagonal kernel
        // value: never positive, zero only for constant fields. This is a
        // property of the estimator's bookkeeping — not a defect — and it
        // is why near-zero field results (e.g. a slightly negative MMD² on
        // an unchanged path) read as "no difference" rather than an error.
        let m = gaussian_mmd_squared(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 1.0).unwrap();
        assert!(m <= 1e-12, "identical arrays must not give positive MMD: {m}");
        assert!(m > -1.0, "unexpected magnitude: {m}");
        // Constant field: every kernel value is 1, diagonal effect cancels.
        assert_eq!(
            gaussian_mmd_squared(&[2.0, 2.0, 2.0], &[2.0, 2.0, 2.0], 1.0),
            Some(0.0)
        );
    }
    #[test]
    fn mmd_is_symmetric_in_its_samples() {
        let x = [0.0, 0.5, 1.0];
        let y = [2.0, 2.5, 3.0];
        let a = gaussian_mmd_squared(&x, &y, 1.0).unwrap();
        let b = gaussian_mmd_squared(&y, &x, 1.0).unwrap();
        assert!((a - b).abs() < 1e-12, "got {a} vs {b}");
    }
    #[test]
    fn quantile_shift_detects_pure_location_shift() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let grid = [0.1, 0.5, 0.9];
        let shifts = quantile_shift(&x, &y, &grid).unwrap();
        for s in shifts {
            assert!((s - 1.0).abs() < 1e-12, "got {s}");
        }
        assert_eq!(quantile_shift(&x, &y, &[]), None);
        assert_eq!(quantile_shift(&x, &y, &[1.5]), None);
        assert_eq!(quantile_shift(&[], &y, &grid), None);
    }
    #[test]
    fn quantile_shift_isolates_tail_contamination() {
        // Same bulk, one hot upper-tail value: lower quantiles ~0,
        // upper quantile carries the shift.
        let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut y = x;
        y[9] = 29.0;
        let shifts = quantile_shift(&x, &y, &[0.1, 0.5, 0.95]).unwrap();
        assert!(shifts[0].abs() < 1e-9, "lower tail moved: {:?}", shifts);
        assert!(shifts[2] > 5.0, "upper tail missed: {:?}", shifts);
    }
    #[test]
    fn variance_shift_is_stronger_in_mmd_energy_than_median() {
        // Symmetric widening: median barely moves, spread statistics fire.
        let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let y = [-6.0, -3.0, 0.0, 3.0, 6.0];
        let med_gap = (audit_core::median(&finite(&y)).unwrap()
            - audit_core::median(&finite(&x)).unwrap())
        .abs();
        let ed = energy_distance(&x, &y).unwrap();
        let bw = pooled_bandwidth(&x, &y).unwrap();
        let mmd = gaussian_mmd_squared(&x, &y, bw).unwrap();
        assert!(med_gap < 1e-12);
        assert!(ed > 1.0, "energy missed variance shift: {ed}");
        assert!(mmd > 0.01, "mmd missed variance shift: {mmd}");
    }
    #[test]
    fn wasserstein1_matches_mean_gap_for_location_shift() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [2.0, 3.0, 4.0, 5.0];
        assert!((wasserstein1(&x, &y).unwrap() - 2.0).abs() < 1e-12);
        assert_eq!(wasserstein1(&x, &x).unwrap(), 0.0);
        assert_eq!(wasserstein1(&[], &x), None);
        // Unequal sizes still integrate the same unit shift.
        let w = wasserstein1(&[0.0, 2.0], &[1.0, 3.0, 5.0]).unwrap();
        assert!((w - 2.0).abs() < 0.6, "got {w}");
    }
    #[test]
    fn pooled_bandwidth_uses_scaled_mad_with_floor() {
        let bw = pooled_bandwidth(&[0.0, 1.0, 2.0], &[10.0, 11.0, 12.0]).unwrap();
        assert!(bw > 1.0, "got {bw}");
        // Degenerate constant field -> epsilon floor, still finite.
        assert_eq!(
            pooled_bandwidth(&[3.0, 3.0], &[3.0, 3.0]),
            Some(BANDWIDTH_EPSILON)
        );
        assert_eq!(pooled_bandwidth(&[], &[]), None);
    }
    #[test]
    fn block_label_swap_exchanges_selected_blocks() {
        let (xs, ys) = block_label_swap(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[0, 0, 1], &[true, false]).unwrap();
        assert_eq!(xs, vec![4.0, 5.0, 3.0]);
        assert_eq!(ys, vec![1.0, 2.0, 6.0]);
        assert_eq!(block_label_swap(&[1.0], &[2.0], &[0], &[true, false]), None);
        assert_eq!(block_label_swap(&[], &[], &[], &[]), None);
    }
    #[test]
    fn null_fixture_stays_inside_simulation_envelope() {
        // Unchanged paired field: observed energy equals every null
        // replicate (swapping identical labels is a no-op), so the
        // Monte Carlo p-value is exactly 1 — the null is retained.
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = x;
        let obs = energy_distance(&x, &y).unwrap();
        assert_eq!(obs, 0.0);
        let mut nulls = Vec::new();
        for seed in 0..19u64 {
            let flips = audit_core::deterministic_flips(2, seed);
            let (nx, ny) = block_label_swap(&x, &y, &[0, 0, 0, 1, 1, 1], &flips).unwrap();
            nulls.push(energy_distance(&nx, &ny).unwrap());
        }
        assert_eq!(permutation_pvalue(obs, &nulls), Some(1.0));
    }
    #[test]
    fn location_shift_rejects_against_pair_swap_null() {
        // Uniform +2 shift, one block per pair: swapping a pair's labels
        // moves its shift to the other sample, so mixed replicates dilute
        // the separation and the observed statistic ranks at the top.
        // (Whole-block swaps of block-homogeneous data cannot dilute a
        // label-symmetric statistic — swapping the shifted block merely
        // moves the shift to the other label — so the null must resolve
        // at least to the pair level to have power here.)
        let x = [0.0, 0.1, -0.1, 0.05, 0.2, -0.2, 0.15, -0.15];
        let y = [2.0, 2.1, 1.9, 2.05, 2.2, 1.8, 2.15, 1.85];
        let blocks = [0usize, 1, 2, 3, 4, 5, 6, 7];
        let obs = energy_distance(&x, &y).unwrap();
        let mut nulls = Vec::new();
        for seed in 0..99u64 {
            let flips = audit_core::deterministic_flips(8, seed);
            let (nx, ny) = block_label_swap(&x, &y, &blocks, &flips).unwrap();
            nulls.push(energy_distance(&nx, &ny).unwrap());
        }
        let p = permutation_pvalue(obs, &nulls).unwrap();
        assert!(p <= 0.1, "shift did not reject: p={p}");
    }
}
