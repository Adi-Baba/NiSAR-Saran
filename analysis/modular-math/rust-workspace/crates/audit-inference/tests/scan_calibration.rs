//! Scan-calibration fixtures: a planted coherent cluster clears a
//! dissolving max-statistic null, while a null field is retained.
//!
//! A whole-block sign-flip null cannot dissolve a block-aligned cluster
//! (every replicate keeps the same regional maximum), so power is shown
//! with a cell-resolving sign-flip null over a multiscale family — the
//! same resolution caveat documented for the distribution module. The
//! block null remains the validity instrument (null fixture below).

use audit_inference::*;

#[test]
fn planted_cluster_clears_max_statistic_null() {
    // 6x6 field; coherent +4 region over the left half (columns 0-2,
    // all rows) on deterministic background. No small fixed region fully
    // contains it except its true support, so the family maximum must be
    // earned by coherent coverage rather than block alignment.
    let (rows, cols) = (6, 6);
    let mut field = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols / 2 {
            field[r * cols + c] = 4.0;
        }
    }
    for (i, v) in field.iter_mut().enumerate() {
        *v += 0.5 * ((1.7 * i as f64).sin());
    }
    // Multiscale family: whole field, left/right halves, nine 2x2 blocks.
    let regions: Vec<Vec<usize>> = {
        let mut r: Vec<Vec<usize>> = vec![
            (0..rows * cols).collect(),
            (0..rows * cols).filter(|i| i % cols < cols / 2).collect(),
            (0..rows * cols).filter(|i| i % cols >= cols / 2).collect(),
        ];
        for br in 0..3 {
            for bc in 0..3 {
                let mut cells = Vec::new();
                for dr in 0..2 {
                    for dc in 0..2 {
                        cells.push((br * 2 + dr) * cols + bc * 2 + dc);
                    }
                }
                r.push(cells);
            }
        }
        r
    };
    let family_max = |f: &[f64]| {
        regions
            .iter()
            .map(|reg| reg.iter().map(|&i| f[i]).sum::<f64>() / (reg.len() as f64).sqrt())
            .fold(0.0, f64::max)
    };
    let observed = family_max(&field);
    // Cell-resolving sign-flip null dissolves the cluster in most replicates.
    let n = field.len();
    let null: Vec<f64> = (0..199u64)
        .map(|s| {
            let flips = block_flip_draw(n, 7, s);
            let rep: Vec<f64> = field
                .iter()
                .enumerate()
                .map(|(i, &v)| if flips[i] { -v } else { v })
                .collect();
            family_max(&rep)
        })
        .collect();
    let p = max_statistic(observed, &null).unwrap();
    assert!(p < 0.05, "planted cluster must clear the scan null: p={p}");
    let t = scan_threshold(&null, 0.05).unwrap();
    assert!(observed > t, "observed {observed} must clear the 5% threshold {t}");
}

#[test]
fn null_field_is_retained_with_p_one_envelope() {
    // All-zero field: every null maximum is zero, observation is zero.
    let null = vec![0.0; 19];
    assert_eq!(max_statistic(0.0, &null), Some(1.0));
    assert_eq!(scan_threshold(&null, 0.05), Some(0.0));
    // Degenerate calibration requests stay uninformative.
    assert_eq!(max_statistic(1.0, &[]), None);
    assert_eq!(group_improvement_p(0.2, &[]), None);
}

#[test]
fn group_null_distinguishes_predictive_from_null_covariate() {
    // Predictive group: observed +40% held-out gain vs null gains near zero.
    let null_gains = vec![0.02, -0.01, 0.03, 0.0, -0.02, 0.01, 0.04, -0.03];
    let p_hit = group_improvement_p(0.40, &null_gains).unwrap();
    assert!(p_hit < 0.2, "predictive group must beat its permutation null: {p_hit}");
    // Null group: observed gain inside its own null -> retained.
    let p_null = group_improvement_p(0.01, &null_gains).unwrap();
    assert!(p_null > 0.2, "null group must be retained: {p_null}");
}
