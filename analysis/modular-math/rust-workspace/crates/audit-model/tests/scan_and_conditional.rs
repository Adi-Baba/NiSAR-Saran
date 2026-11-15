//! Paper 3/4 fixtures: graph-scan regions rank a planted coherent cluster
//! above an isolated spike, and nested conditional fits recover a
//! predictive covariate group without promoting a null group.

use audit_model::*;

fn cluster_grid() -> (Vec<f64>, Vec<bool>) {
    // 8x8: valid top-left 4x4 patch, valid bottom-right 4x4 patch,
    // invalid moat between them -> exactly two rook components.
    let (rows, cols) = (8, 8);
    let mut values = vec![0.0; rows * cols];
    let mut valid = vec![false; rows * cols];
    for r in 0..4 {
        for c in 0..4 {
            valid[r * cols + c] = true;
        }
    }
    for r in 5..8 {
        for c in 5..8 {
            valid[r * cols + c] = true;
        }
    }
    // Coherent +2 cluster: 2x2 at rows 1-2, cols 1-2 (top-left patch).
    for r in 1..3 {
        for c in 1..3 {
            values[r * cols + c] = 2.0;
        }
    }
    // Isolated +2 spike in the bottom-right patch.
    values[7 * cols + 7] = 2.0;
    (values, valid)
}

#[test]
fn scan_ranks_coherent_cluster_above_isolated_spike() {
    let (rows, cols) = (8, 8);
    let (values, valid) = cluster_grid();
    let comps = connected_components(&valid, rows, cols, false).unwrap();
    assert_eq!(comps.len(), 2);
    let stats: Vec<RegionStat> = comps
        .iter()
        .map(|c| region_stats(&values, c, rows, cols).unwrap())
        .collect();
    // Cluster patch: 4 cells at 2.0 -> score 8/sqrt(16); spike patch lower.
    assert!(stats[0].score > stats[1].score, "{stats:?}");
    assert_eq!(stats[0].area, 16);
    assert!((stats[0].score - 8.0 / 4.0).abs() < 1e-12);
    // Centroid of the top-left patch is its center.
    assert!((stats[0].centroid_row - 1.5).abs() < 1e-12);
    assert!((stats[0].centroid_col - 1.5).abs() < 1e-12);
    // The coherent patch is smoother than a checkerboard of equal amplitude.
    let mut checker = vec![0.0; 16];
    for r in 0..4 {
        for c in 0..4 {
            checker[r * 4 + c] = if (r + c) % 2 == 0 { 1.0 } else { -1.0 };
        }
    }
    let all: Vec<usize> = (0..16).collect();
    let r_cluster = region_stats(&values[0..16].to_vec(), &all, 4, 4).unwrap();
    let r_checker = region_stats(&checker, &all, 4, 4).unwrap();
    assert!(r_checker.roughness > r_cluster.roughness);
}

#[test]
fn connectivity_and_geometry_reject_degenerate_input() {
    assert_eq!(connected_components(&[true; 4], 2, 3, false), None);
    assert_eq!(connected_components(&[], 0, 0, false), None);
    // Diagonal-only contact: split under rook, joined under queen.
    let valid = vec![true, false, false, true];
    assert_eq!(connected_components(&valid, 2, 2, false).unwrap().len(), 2);
    assert_eq!(connected_components(&valid, 2, 2, true).unwrap().len(), 1);
    // All-nonfinite region is not a candidate.
    assert_eq!(region_stats(&[f64::NAN; 4], &[0, 1, 2, 3], 2, 2), None);
    assert_eq!(region_stats(&[1.0; 4], &[], 2, 2), None);
    // Single cell: perimeter 4, roughness 0.
    let single = region_stats(&[3.0], &[0], 1, 1).unwrap();
    assert_eq!((single.perimeter, single.roughness, single.score), (4, 0.0, 3.0));
}

#[test]
fn multiscale_tiling_covers_without_padding() {
    let tiles = multiscale_tiles(5, 5, &[2, 4]);
    // size 2: 3x3 = 9 tiles; size 4: 2x2 = 4 tiles.
    assert_eq!(tiles.len(), 13);
    assert!(tiles.iter().all(|t| t.row < 5 && t.col < 5));
    let edge = tile_cells(Tile { row: 4, col: 4, size: 2 }, 5, 5);
    assert_eq!(edge, vec![24]);
    assert!(multiscale_tiles(5, 5, &[0]).is_empty());
}

#[test]
fn conditional_fit_recovers_predictive_group_not_null_group() {
    // y = 1 + 2*q over 24 rows in 4 blocks; null column is alternating noise.
    let n = 24;
    let q: Vec<f64> = (0..n).map(|i| (i % 6) as f64).collect();
    let z: Vec<f64> = (0..n).map(|i| ((i * 7) % 3) as f64).collect();
    let y: Vec<f64> = q.iter().map(|v| 1.0 + 2.0 * v).collect();
    let blocks: Vec<usize> = (0..n).map(|i| i / 6).collect();
    let (train, test) = block_split(&blocks, 4, &[3]).unwrap();
    assert_eq!((train.len(), test.len()), (18, 6));
    let gather = |v: &[f64], idx: &[usize]| idx.iter().map(|&i| v[i]).collect::<Vec<_>>();
    // Baseline: intercept-only mean of train.
    let train_y = gather(&y, &train);
    let base = train_y.iter().sum::<f64>() / train_y.len() as f64;
    let test_y = gather(&y, &test);
    let (base_mae, _) = held_out_errors(&test_y, &vec![base; test.len()]).unwrap();
    // Full model with the predictive group.
    let fit = huber_regression(&[gather(&q, &train)], &train_y, 10, 1.345).unwrap();
    assert!((fit.coefficients[1] - 2.0).abs() < 1e-9);
    let pred: Vec<f64> = test.iter().map(|&i| predict_row(&fit.coefficients, &[q.clone()], i).unwrap()).collect();
    let (full_mae, _) = held_out_errors(&test_y, &pred).unwrap();
    let good = error_reduction(base_mae, full_mae).unwrap();
    assert!(good > 0.9, "predictive group should explain the shift: {good}");
    // Null group: no improvement over baseline.
    let fit_z = huber_regression(&[gather(&z, &train)], &train_y, 10, 1.345).unwrap();
    let pred_z: Vec<f64> = test.iter().map(|&i| predict_row(&fit_z.coefficients, &[z.clone()], i).unwrap()).collect();
    let (z_mae, _) = held_out_errors(&test_y, &pred_z).unwrap();
    let null_imp = error_reduction(base_mae, z_mae).unwrap();
    assert!(null_imp < good, "null group must not beat the predictive group");
    // Collinearity report flags the duplicated column.
    let flags = collinear_flags(&[q.clone(), q.clone(), z.clone()], 0.99);
    assert_eq!(flags.len(), 1);
    assert_eq!((flags[0].0, flags[0].1), (0, 1));
    // Missingness accounting: one NaN row excluded, never imputed.
    let mut y_gap = y.clone();
    y_gap[0] = f64::NAN;
    assert_eq!(complete_rows(&y_gap, &[q, z]).len(), n - 1);
}

#[test]
fn block_split_and_errors_reject_degenerate_input() {
    assert_eq!(block_split(&[], 2, &[0]), None);
    assert_eq!(block_split(&[0, 1], 2, &[5]), None); // held-out id out of range
    assert_eq!(block_split(&[0, 0], 2, &[0, 1]), None); // empty train side
    assert_eq!(held_out_errors(&[1.0], &[1.0, 2.0]), None);
    assert_eq!(held_out_errors(&[f64::NAN], &[f64::NAN]), None);
    assert_eq!(error_reduction(0.0, 0.0), None);
    assert_eq!(column_correlation(&[1.0, 1.0], &[1.0, 2.0]), None);
    assert_eq!(predict_row(&[1.0], &[vec![1.0]], 5), None);
}
