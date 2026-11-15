//! Paper 2 synthetic fixtures: location shift, variance-only shift, tail
//! contamination, and an unchanged paired null — with the expected
//! signature pattern from `papers/paper02_distribution_shift.md` §5.

use audit_core::median;
use audit_distribution::{
    block_label_swap, energy_distance, gaussian_mmd_squared, permutation_pvalue,
    pooled_bandwidth, quantile_shift, wasserstein1,
};

const GRID: [f64; 5] = [0.05, 0.25, 0.5, 0.75, 0.95];

#[test]
fn location_shift_moves_every_summary() {
    let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let y = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let med_gap =
        median(&y).unwrap() - median(&x).unwrap();
    assert!((med_gap - 2.0).abs() < 1e-12);
    for s in quantile_shift(&x, &y, &GRID).unwrap() {
        assert!((s - 2.0).abs() < 1e-9, "got {s}");
    }
    // Shift (2.0) below the sample spread: energy grows quadratically
    // near zero, so it reads ~0.5 here while Wasserstein reads the full 2.
    assert!(energy_distance(&x, &y).unwrap() > 0.2);
    let bw = pooled_bandwidth(&x, &y).unwrap();
    assert!(gaussian_mmd_squared(&x, &y, bw).unwrap() > 0.01);
    assert!((wasserstein1(&x, &y).unwrap() - 2.0).abs() < 1e-9);
}

#[test]
fn variance_shift_fires_spread_statistics_not_median() {
    let x = [-2.0, -1.0, 0.0, 1.0, 2.0, -1.5, 0.5, 1.5];
    let y = [-6.0, -3.0, 0.0, 3.0, 6.0, -4.5, 1.5, 4.5];
    let med_gap = (median(&y).unwrap() - median(&x).unwrap()).abs();
    assert!(med_gap < 0.6, "median moved: {med_gap}");
    let ed = energy_distance(&x, &y).unwrap();
    let bw = pooled_bandwidth(&x, &y).unwrap();
    let mmd = gaussian_mmd_squared(&x, &y, bw).unwrap();
    assert!(ed > 1.0, "energy missed it: {ed}");
    assert!(mmd > 0.01, "mmd missed it: {mmd}");
    // Spread shift displaces the outer quantiles more than the median.
    let shifts = quantile_shift(&x, &y, &GRID).unwrap();
    assert!(
        shifts[0].abs() + shifts[4].abs() > shifts[2].abs(),
        "tails should dominate: {shifts:?}"
    );
}

#[test]
fn tail_contamination_concentrates_in_upper_quantiles() {
    let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mut y = x;
    y[9] = 29.0;
    let shifts = quantile_shift(&x, &y, &GRID).unwrap();
    assert!(shifts[0].abs() < 1.0, "lower quantiles moved: {shifts:?}");
    assert!(shifts[4] > 3.0, "upper quantile missed: {shifts:?}");
    assert!(energy_distance(&x, &y).unwrap() > 0.0);
    assert!(wasserstein1(&x, &y).unwrap() > 0.0);
}

#[test]
fn unchanged_null_survives_with_full_pipeline() {
    // Identical paired samples across two blocks: every statistic is at
    // its null value and every permutation p-value is 1.
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = x;
    let blocks = [0usize, 0, 0, 0, 1, 1, 1, 1];
    let bw = pooled_bandwidth(&x, &y).unwrap();
    for stat in [energy_distance(&x, &y).unwrap(), wasserstein1(&x, &y).unwrap()] {
        assert!(stat.abs() < 1e-9, "null stat not zero: {stat}");
    }
    // MMD on literally identical arrays sits at its non-positive diagonal
    // value (see unit test), still far below any shifted alternative.
    let m0 = gaussian_mmd_squared(&x, &y, bw).unwrap();
    assert!(m0 <= 1e-12, "null mmd positive: {m0}");
    let mut nulls = Vec::new();
    for seed in 0..19u64 {
        let flips = audit_core::deterministic_flips(2, seed);
        let (nx, ny) = block_label_swap(&x, &y, &blocks, &flips).unwrap();
        nulls.push(energy_distance(&nx, &ny).unwrap());
    }
    assert_eq!(permutation_pvalue(0.0, &nulls), Some(1.0));
}
