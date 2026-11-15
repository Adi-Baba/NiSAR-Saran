//! Spatial-block and graph-scan primitives.
//!
//! Group A owns this crate (papers 1–2): block geometry, neighbour-graph
//! energies, fixed-window scans, and spatial autocorrelation (Moran's I).
//! All functions are deterministic and dependency-free. Non-finite cells
//! are skipped pairwise — never imputed — and windows with no valid
//! support are not candidates.

pub fn scan_score(values: &[f64], region: &[usize]) -> Option<f64> {
    if region.is_empty() {
        return None;
    }
    let mut sum = 0.0;
    let mut support = 0usize;
    for i in region {
        match values.get(*i) {
            Some(x) if x.is_finite() => {
                sum += x;
                support += 1;
            }
            _ => {}
        }
    }
    if support == 0 {
        return None;
    }
    Some(sum / (support as f64).sqrt())
}

pub fn laplacian_energy(values: &[f64], rows: usize, cols: usize) -> Option<f64> {
    if rows.checked_mul(cols)? != values.len() || rows == 0 || cols == 0 {
        return None;
    }
    let mut total = 0.0;
    for r in 0..rows {
        for c in 0..cols {
            let i = r * cols + c;
            let v = values[i];
            if !v.is_finite() {
                continue;
            }
            if c + 1 < cols && values[i + 1].is_finite() {
                total += (v - values[i + 1]).powi(2);
            }
            if r + 1 < rows && values[i + cols].is_finite() {
                total += (v - values[i + cols]).powi(2);
            }
        }
    }
    Some(total)
}

/// Moran's I with 4-neighbour (rook) weights on a row-major grid.
///
/// `I = (N / W) * Σ_ij w_ij (x_i - x̄)(x_j - x̄) / Σ_i (x_i - x̄)²`,
/// where each undirected neighbour pair contributes two directed edges
/// (`W = 2E`). Pairs touching a non-finite cell are skipped, and `N` is
/// the finite count. Clustered fields give positive `I`; a checkerboard
/// gives negative `I`; the null expectation is `-1/(N-1)`.
///
/// Returns `None` when the grid shape mismatches, fewer than two cells
/// are finite, the field is constant (zero variance — autocorrelation is
/// undefined, not zero), or no neighbour pair is jointly finite.
pub fn morans_i(values: &[f64], rows: usize, cols: usize) -> Option<f64> {
    if rows.checked_mul(cols)? != values.len() || rows == 0 || cols == 0 {
        return None;
    }
    let finite_idx: Vec<usize> = values
        .iter()
        .enumerate()
        .filter_map(|(i, x)| x.is_finite().then_some(i))
        .collect();
    let n = finite_idx.len();
    if n < 2 {
        return None;
    }
    let mean = finite_idx.iter().map(|&i| values[i]).sum::<f64>() / n as f64;
    let denom: f64 = finite_idx.iter().map(|&i| (values[i] - mean).powi(2)).sum();
    if !(denom > 0.0) {
        return None;
    }
    // Right and down edges; each counted in both directions.
    let mut cross = 0.0;
    let mut edges = 0usize;
    for &i in &finite_idx {
        let r = i / cols;
        let c = i % cols;
        if c + 1 < cols && values[i + 1].is_finite() {
            cross += (values[i] - mean) * (values[i + 1] - mean);
            edges += 1;
        }
        if r + 1 < rows && values[i + cols].is_finite() {
            cross += (values[i] - mean) * (values[i + cols] - mean);
            edges += 1;
        }
    }
    if edges == 0 {
        return None;
    }
    let w = 2.0 * edges as f64;
    Some((n as f64 / w) * (2.0 * cross) / denom)
}

/// Approximate effective sample size under AR(1)-like autocorrelation.
///
/// `n_eff = n * (1 - ρ) / (1 + ρ)` with `ρ` clipped to `[0, 0.99)`.
/// Negative autocorrelation estimates map to `n` (no deflation without
/// evidence of positive dependence). A reporting aid for uncertainty
/// captions, never a substitute for the block null.
pub fn effective_sample_size(n: usize, rho: f64) -> Option<usize> {
    if n == 0 || !rho.is_finite() {
        return None;
    }
    let r = rho.clamp(0.0, 0.99);
    Some(((n as f64 * (1.0 - r) / (1.0 + r)).round() as usize).max(1))
}

/// Deterministic block partition of a row-major grid.
///
/// Block `(br, bc)` covers rows `br*block_rows..(br+1)*block_rows` (last
/// block truncated at the edge) and likewise for columns; ids are
/// row-major over blocks. `None` for degenerate geometry. Every cell —
/// valid or not — gets a block so the null geometry never depends on the
/// data mask.
pub fn block_ids(
    rows: usize,
    cols: usize,
    block_rows: usize,
    block_cols: usize,
) -> Option<Vec<usize>> {
    if rows.checked_mul(cols).is_none()
        || rows == 0
        || cols == 0
        || block_rows == 0
        || block_cols == 0
    {
        return None;
    }
    let n_block_cols = cols.div_ceil(block_cols);
    let mut ids = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            ids.push((r / block_rows) * n_block_cols + (c / block_cols));
        }
    }
    Some(ids)
}

/// Number of blocks in a [`block_ids`] layout.
pub fn block_count(ids: &[usize]) -> usize {
    ids.iter().max().map(|m| m + 1).unwrap_or(0)
}

/// Block sign-flip null replicate: negate every cell whose block is
/// selected in `flips`. Complete blocks move together, preserving local
/// dependence. Non-finite cells pass through unchanged. `None` on
/// length mismatch or empty input.
pub fn block_sign_flip(
    values: &[f64],
    blocks: &[usize],
    flips: &[bool],
) -> Option<Vec<f64>> {
    if values.len() != blocks.len() || values.is_empty() {
        return None;
    }
    if block_count(blocks) != flips.len() {
        return None;
    }
    Some(
        values
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if x.is_finite() && flips[blocks[i]] {
                    -x
                } else {
                    x
                }
            })
            .collect(),
    )
}

/// Monte Carlo p-value on the window-score magnitude:
/// `(1 + #{|T_b| >= |T_obs|}) / (1 + B)`. `None` for non-finite
/// observation or no finite null replicate.
pub fn window_pvalue(observed: f64, null_scores: &[f64]) -> Option<f64> {
    if !observed.is_finite() {
        return None;
    }
    let finite_null: Vec<f64> = null_scores
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    if finite_null.is_empty() {
        return None;
    }
    let hits = finite_null
        .iter()
        .filter(|x| x.abs() >= observed.abs())
        .count();
    Some((1 + hits) as f64 / (1 + finite_null.len()) as f64)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowScan {
    pub row: usize,
    pub col: usize,
    pub rows: usize,
    pub cols: usize,
    pub score: f64,
}

/// Exhaustive fixed-window scan on a row-major grid.
///
/// The window dimensions are inputs fixed before ranking. Non-finite cells are
/// ignored in the sum and support normalization; an all-invalid window is not a
/// candidate. The returned score keeps its sign while ranking uses magnitude.
pub fn max_fixed_window_scan(
    values: &[f64],
    rows: usize,
    cols: usize,
    window_rows: usize,
    window_cols: usize,
) -> Option<WindowScan> {
    if rows.checked_mul(cols)? != values.len()
        || rows == 0
        || cols == 0
        || window_rows == 0
        || window_cols == 0
        || window_rows > rows
        || window_cols > cols
    {
        return None;
    }
    let mut best: Option<WindowScan> = None;
    for row in 0..=(rows - window_rows) {
        for col in 0..=(cols - window_cols) {
            let mut sum = 0.0;
            let mut support = 0usize;
            for rr in row..(row + window_rows) {
                for cc in col..(col + window_cols) {
                    let value = values[rr * cols + cc];
                    if value.is_finite() {
                        sum += value;
                        support += 1;
                    }
                }
            }
            if support == 0 {
                continue;
            }
            let score = sum / (support as f64).sqrt();
            if best.map(|b| score.abs() > b.score.abs()).unwrap_or(true) {
                best = Some(WindowScan {
                    row,
                    col,
                    rows: window_rows,
                    cols: window_cols,
                    score,
                });
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scan_score_normalizes_region_size() {
        let s = scan_score(&[2.0, 2.0, 0.0], &[0, 1]).unwrap();
        assert!((s - 2.0 * 2.0_f64.sqrt()).abs() < 1e-12);
    }
    #[test]
    fn scan_score_skips_nonfinite_and_rejects_empty_support() {
        assert_eq!(scan_score(&[f64::NAN, 1.0], &[0]), None);
        assert_eq!(scan_score(&[1.0], &[]), None);
        assert_eq!(scan_score(&[1.0], &[5]), None);
    }
    #[test]
    fn constant_grid_has_zero_laplacian_energy() {
        assert_eq!(laplacian_energy(&[1.0; 4], 2, 2), Some(0.0));
    }
    #[test]
    fn laplacian_ignores_nonfinite_pairs() {
        assert_eq!(laplacian_energy(&[1.0, f64::NAN, 1.0, 1.0], 2, 2), Some(0.0));
        assert_eq!(laplacian_energy(&[1.0; 3], 2, 2), None);
    }
    #[test]
    fn fixed_window_finds_planted_cluster() {
        let mut values = vec![0.0; 16];
        for index in [5, 6, 9, 10] {
            values[index] = 2.0;
        }
        let scan = max_fixed_window_scan(&values, 4, 4, 2, 2).unwrap();
        assert_eq!((scan.row, scan.col), (1, 1));
        assert!((scan.score - 4.0).abs() < 1e-12);
    }
    #[test]
    fn fixed_window_rejects_degenerate_geometry() {
        assert_eq!(max_fixed_window_scan(&[1.0; 4], 2, 2, 3, 1), None);
        assert_eq!(max_fixed_window_scan(&[1.0; 4], 2, 2, 0, 1), None);
        assert_eq!(max_fixed_window_scan(&[f64::NAN; 4], 2, 2, 2, 2), None);
    }
    #[test]
    fn morans_i_is_positive_for_cluster_negative_for_checkerboard() {
        // One hot corner block on 4x4: clustered -> positive I.
        let mut clustered = vec![0.0; 16];
        for index in [0, 1, 4, 5] {
            clustered[index] = 4.0;
        }
        let i = morans_i(&clustered, 4, 4).unwrap();
        assert!(i > 0.2, "clustered I too small: {i}");
        // Checkerboard on 4x4: anti-correlated -> negative I.
        let mut board = vec![0.0; 16];
        for r in 0..4 {
            for c in 0..4 {
                board[r * 4 + c] = if (r + c) % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
        let j = morans_i(&board, 4, 4).unwrap();
        assert!(j < -0.2, "checkerboard I not negative: {j}");
    }
    #[test]
    fn morans_i_rejects_degenerate_fields() {
        assert_eq!(morans_i(&[1.0; 4], 2, 2), None); // constant
        assert_eq!(morans_i(&[1.0], 1, 1), None); // single cell
        assert_eq!(morans_i(&[1.0, f64::NAN, f64::NAN, f64::NAN], 2, 2), None);
        assert_eq!(morans_i(&[1.0; 3], 2, 2), None); // shape mismatch
    }
    #[test]
    fn block_ids_tile_rectangles_deterministically() {
        // 4x4 grid, 2x2 blocks -> ids 0,1 / 2,3 quadrants.
        let ids = block_ids(4, 4, 2, 2).unwrap();
        assert_eq!(block_count(&ids), 4);
        assert_eq!(ids[0], 0);
        assert_eq!(ids[2], 1);
        assert_eq!(ids[8], 2);
        assert_eq!(ids[10], 3);
        // Ragged edge: 3x3 grid with 2x2 blocks still covers all 9 cells.
        let ragged = block_ids(3, 3, 2, 2).unwrap();
        assert_eq!(ragged.len(), 9);
        assert_eq!(block_count(&ragged), 4);
        assert_eq!(block_ids(0, 4, 2, 2), None);
        assert_eq!(block_ids(4, 4, 0, 2), None);
    }
    #[test]
    fn block_sign_flip_moves_whole_blocks() {
        let ids = block_ids(2, 4, 2, 2).unwrap(); // two blocks: cols 0-1, 2-3
        let out = block_sign_flip(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &ids, &[true, false]).unwrap();
        assert_eq!(out, vec![-1.0, -2.0, 3.0, 4.0, -5.0, -6.0, 7.0, 8.0]);
        // Non-finite passes through unflipped.
        let out2 = block_sign_flip(&[f64::NAN, 1.0], &[0, 0], &[true]).unwrap();
        assert!(out2[0].is_nan() && out2[1] == -1.0);
        assert_eq!(block_sign_flip(&[1.0], &[0, 0], &[true]), None);
        assert_eq!(block_sign_flip(&[1.0], &[0], &[true, false]), None);
    }
    #[test]
    fn window_pvalue_never_returns_zero() {
        assert_eq!(window_pvalue(5.0, &[1.0, 2.0, 3.0]), Some(0.25));
        assert_eq!(window_pvalue(f64::NAN, &[1.0]), None);
        assert_eq!(window_pvalue(1.0, &[]), None);
    }
    #[test]
    fn effective_sample_size_deflates_only_for_positive_rho() {
        assert_eq!(effective_sample_size(100, 0.0), Some(100));
        assert!(effective_sample_size(100, 0.5).unwrap() < 100);
        assert_eq!(effective_sample_size(100, -0.5), Some(100));
        assert_eq!(effective_sample_size(0, 0.5), None);
        assert_eq!(effective_sample_size(100, f64::NAN), None);
    }
}
