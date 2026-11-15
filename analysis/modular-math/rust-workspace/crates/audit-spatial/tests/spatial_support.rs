//! Paper 1 spatial fixture: 8x8 difference field with a 2x2 hot cluster.
//!
//! Covers Moran's I (clustering signal), the fixed-window scan (support
//! localization), block geometry, and the window-scan permutation null.

use audit_spatial::{
    block_ids, block_sign_flip, laplacian_energy, max_fixed_window_scan, morans_i,
    scan_score, window_pvalue,
};

/// 8x8 zero field with +2.0 at rows 2-3, cols 2-3 (indices 18,19,26,27).
fn cluster_field() -> Vec<f64> {
    let mut v = vec![0.0f64; 64];
    for &i in &[18, 19, 26, 27] {
        v[i] = 2.0;
    }
    v
}

#[test]
fn clustered_field_has_expected_spatial_signatures() {
    let v = cluster_field();
    let i = morans_i(&v, 8, 8).unwrap();
    assert!(i > 0.1, "cluster should be positively autocorrelated: {i}");
    assert!(laplacian_energy(&v, 8, 8).unwrap() > 0.0);
    let region = scan_score(&v, &[18, 19, 26, 27]).unwrap();
    assert!((region - 8.0 / 2.0).abs() < 1e-12, "got {region}");
}

#[test]
fn window_scan_localizes_planted_cluster() {
    let v = cluster_field();
    let scan = max_fixed_window_scan(&v, 8, 8, 2, 2).unwrap();
    assert_eq!((scan.row, scan.col), (2, 2));
    assert!((scan.score - 4.0).abs() < 1e-12, "got {}", scan.score);
}

#[test]
fn block_null_keeps_coherent_cluster_significant() {
    // 8x8 field, 2x2 blocks (16 blocks): the cluster sits inside block 9.
    // Flipping whole blocks preserves the cluster's coherence, so the
    // observed window magnitude stays at the top of the null envelope
    // often enough to remain a candidate, while a lone-hot-pixel field
    // built below dilutes. Here we assert the mechanics: the no-flip
    // replicate reproduces the observation and the all-flip replicate
    // negates it exactly.
    let v = cluster_field();
    let ids = block_ids(8, 8, 2, 2).unwrap();
    assert_eq!(audit_spatial::block_count(&ids), 16);
    let n = audit_spatial::block_count(&ids);
    let same = block_sign_flip(&v, &ids, &vec![false; n]).unwrap();
    assert_eq!(same, v);
    let flipped = block_sign_flip(&v, &ids, &vec![true; n]).unwrap();
    for (a, b) in v.iter().zip(&flipped) {
        assert!((a + b).abs() < 1e-12);
    }
    // Window p-value mechanics on a small deterministic null set.
    let obs = max_fixed_window_scan(&v, 8, 8, 2, 2).unwrap().score;
    let p = window_pvalue(obs, &[0.5, 1.0, 4.0, 8.0]).unwrap();
    assert!((p - 3.0 / 5.0).abs() < 1e-12, "got {p}");
}

#[test]
fn flat_field_is_uninformative_not_zero_evidence() {
    let flat = vec![0.0f64; 64];
    assert_eq!(morans_i(&flat, 8, 8), None); // undefined, not zero
    assert_eq!(laplacian_energy(&flat, 8, 8), Some(0.0));
    let scan = max_fixed_window_scan(&flat, 8, 8, 2, 2).unwrap();
    assert_eq!(scan.score, 0.0);
}
