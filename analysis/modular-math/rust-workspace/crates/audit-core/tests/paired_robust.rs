//! Paper 1 synthetic fixture: 8x8 paired field with a four-cell positive
//! cluster, unchanged background, one invalid cell, and one mask transition.
//!
//!mirrors `papers/paper01_paired_spatial_change.md` §6: the test asserts the
//! expected median, MAD, sign fraction, and cluster support end to end.

use audit_core::{paired_delta, paired_summary, permutation_pvalue, sign_flip_mean};

/// Builds the 8x8 paper-1 fixture. Returns `(before, after)` row-major.
fn fixture() -> (Vec<f64>, Vec<f64>) {
    let mut before = vec![1.0f64; 64];
    let mut after = vec![1.0f64; 64];
    // Four-cell positive cluster (+2) at rows 2-3, cols 2-3.
    for &i in &[18, 19, 26, 27] {
        after[i] = 3.0;
    }
    // One invalid cell (NaN before) and one mask transition (NaN after).
    before[0] = f64::NAN;
    after[63] = f64::NAN;
    (before, after)
}

#[test]
fn fixture_reports_expected_robust_summaries() {
    let (before, after) = fixture();
    let s = paired_summary(&before, &after);
    assert_eq!(s.n_pairs, 64);
    assert_eq!(s.finite_before, 63);
    assert_eq!(s.finite_after, 63);
    // Pairs finite on both sides: all but cells 0 and 63.
    assert_eq!(s.finite_pairs, 62);
    // 58 zeros + four 2.0s: median 0, MAD 0, sign fraction 4/62.
    assert_eq!(s.median_delta, Some(0.0));
    assert_eq!(s.mad_delta, Some(0.0));
    assert_eq!(s.robust_effect, Some(0.0));
    let pf = s.positive_fraction.unwrap();
    assert!((pf - 4.0 / 62.0).abs() < 1e-12, "got {pf}");
    // Type-7 q95: position 0.95*61 = 57.95 straddles the last zero (index
    // 57) and first 2.0 (index 58): 0 + 2*0.95 = 1.9.
    let q95 = s.q95_delta.unwrap();
    assert!((q95 - 1.9).abs() < 1e-9, "got {q95}");
}

#[test]
fn cluster_cells_carry_the_only_signal() {
    let (before, after) = fixture();
    let deltas = paired_delta(&before, &after);
    assert_eq!(deltas.len(), 62);
    let hot: Vec<f64> = deltas.into_iter().filter(|&d| d > 0.0).collect();
    assert_eq!(hot, vec![2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn sign_flip_null_exonerates_median_zero_field() {
    // Median-zero fixture: sign-flipped means scatter around 0, so the
    // observed mean sits inside the null envelope (p well above 0.05).
    let (before, after) = fixture();
    let deltas = paired_delta(&before, &after);
    let obs = sign_flip_mean(&deltas, &vec![false; deltas.len()]).unwrap();
    let mut nulls = Vec::new();
    for seed in 0..99u64 {
        let flips = audit_core::deterministic_flips(deltas.len(), seed);
        nulls.push(sign_flip_mean(&deltas, &flips).unwrap());
    }
    let p = permutation_pvalue(obs.abs(), &nulls.iter().map(|x| x.abs()).collect::<Vec<_>>()).unwrap();
    assert!(p > 0.05, "median-zero field rejected: p={p}");
}

#[test]
fn shifted_fixture_rejects_against_sign_flip_null() {
    // Uniform +1 shift over 20 finite pairs: the observed mean exceeds
    // nearly every sign-flip replicate.
    let before = vec![0.0f64; 20];
    let after = vec![1.0f64; 20];
    let deltas = paired_delta(&before, &after);
    let obs = sign_flip_mean(&deltas, &vec![false; deltas.len()]).unwrap();
    assert_eq!(obs, 1.0);
    let mut nulls = Vec::new();
    for seed in 0..99u64 {
        let flips = audit_core::deterministic_flips(deltas.len(), seed);
        nulls.push(sign_flip_mean(&deltas, &flips).unwrap());
    }
    let p = permutation_pvalue(obs, &nulls).unwrap();
    assert!(p < 0.05, "uniform shift retained: p={p}");
}
