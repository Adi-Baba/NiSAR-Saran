//! Small robust regression primitive for conditional explanation.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineFit { pub intercept: f64, pub slope: f64, pub residual_scale: f64, pub n: usize }

#[derive(Debug, Clone, PartialEq)]
pub struct RobustFit {
    /// Intercept followed by one coefficient for each feature column.
    pub coefficients: Vec<f64>,
    pub residual_scale: f64,
    pub n: usize,
}

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for pivot in 0..n {
        let best = (pivot..n).max_by(|i, j| a[*i][pivot].abs().total_cmp(&a[*j][pivot].abs()))?;
        if a[best][pivot].abs() < 1e-12 { return None; }
        a.swap(pivot, best); b.swap(pivot, best);
        let divisor = a[pivot][pivot];
        for col in pivot..n { a[pivot][col] /= divisor; }
        b[pivot] /= divisor;
        for row in 0..n {
            if row == pivot { continue; }
            let factor = a[row][pivot];
            for col in pivot..n { a[row][col] -= factor * a[pivot][col]; }
            b[row] -= factor * b[pivot];
        }
    }
    Some(b)
}

fn median_absolute(values: &[f64]) -> f64 {
    let mut v: Vec<f64> = values.iter().map(|x| x.abs()).filter(|x| x.is_finite()).collect();
    if v.is_empty() { return 0.0; }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Multivariate Huber regression with an intercept and deterministic IRLS.
pub fn huber_regression(features: &[Vec<f64>], y: &[f64], iterations: usize, tuning: f64) -> Option<RobustFit> {
    if features.is_empty() || y.len() < 3 || tuning <= 0.0 || features.iter().any(|column| column.len() != y.len()) { return None; }
    let valid: Vec<usize> = (0..y.len()).filter(|i| y[*i].is_finite() && features.iter().all(|column| column[*i].is_finite())).collect();
    let p = features.len() + 1;
    if valid.len() <= p { return None; }
    let mut weights = vec![1.0; valid.len()];
    let mut coefficients = vec![0.0; p];
    for _ in 0..iterations.max(1) {
        let mut normal = vec![vec![0.0; p]; p];
        let mut rhs = vec![0.0; p];
        for (position, index) in valid.iter().enumerate() {
            let mut row = Vec::with_capacity(p);
            row.push(1.0);
            row.extend(features.iter().map(|column| column[*index]));
            let w = weights[position];
            for i in 0..p {
                rhs[i] += w * row[i] * y[*index];
                for j in 0..p { normal[i][j] += w * row[i] * row[j]; }
            }
        }
        for (i, row) in normal.iter_mut().enumerate() { row[i] += 1e-10; }
        coefficients = solve(normal, rhs)?;
        let residuals: Vec<f64> = valid.iter().map(|index| {
            y[*index] - coefficients[0] - features.iter().enumerate().map(|(j, column)| coefficients[j + 1] * column[*index]).sum::<f64>()
        }).collect();
        let scale = (1.4826 * median_absolute(&residuals)).max(1e-12);
        for (weight, residual) in weights.iter_mut().zip(residuals) {
            let standardized = residual.abs() / scale;
            *weight = if standardized <= tuning { 1.0 } else { tuning / standardized };
        }
    }
    let residuals: Vec<f64> = valid.iter().map(|index| {
        y[*index] - coefficients[0] - features.iter().enumerate().map(|(j, column)| coefficients[j + 1] * column[*index]).sum::<f64>()
    }).collect();
    let residual_scale = (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64).sqrt();
    Some(RobustFit { coefficients, residual_scale, n: valid.len() })
}

pub fn huber_line(x: &[f64], y: &[f64], iterations: usize, tuning: f64) -> Option<LineFit> {
    if x.len() != y.len() || x.len() < 3 || tuning <= 0.0 { return None; }
    let mut w = vec![1.0; x.len()];
    // Theil-Sen initialization prevents a high-leverage outlier from steering
    // the first weighted least-squares step.
    let mut slopes = Vec::new();
    for i in 0..x.len() { for j in (i + 1)..x.len() {
        if x[i].is_finite() && x[j].is_finite() && y[i].is_finite() && y[j].is_finite() && x[j] != x[i] {
            slopes.push((y[j] - y[i]) / (x[j] - x[i]));
        }
    }}
    if slopes.is_empty() { return None; }
    slopes.sort_by(f64::total_cmp);
    let mut b = slopes[slopes.len() / 2];
    let mut intercepts: Vec<f64> = x.iter().zip(y).filter_map(|(x,y)| (x.is_finite()&&y.is_finite()).then_some(y-b*x)).collect();
    intercepts.sort_by(f64::total_cmp);
    let mut a = intercepts[intercepts.len() / 2];
    for i in 0..x.len() {
        let r = y[i] - a - b * x[i];
        let z = r.abs() / tuning;
        w[i] = if z <= 1.0 { 1.0 } else { 1.0 / z };
    }
    // One guarded reweighted solve is deliberately used here. Repeated IRLS
    // without a fixed robust scale can re-admit a high-leverage outlier.
    let _ = iterations;
    for _ in 0..1 {
        let mut sw=0.0; let mut sx=0.0; let mut sy=0.0; let mut sxx=0.0; let mut sxy=0.0;
        for i in 0..x.len() { if x[i].is_finite() && y[i].is_finite() { let q=w[i]; sw+=q; sx+=q*x[i]; sy+=q*y[i]; sxx+=q*x[i]*x[i]; sxy+=q*x[i]*y[i]; }}
        let det = sw*sxx - sx*sx; if det.abs() < 1e-12 { return None; }
        a = (sy*sxx - sx*sxy)/det; b = (sw*sxy - sx*sy)/det;
        let mut abs_residuals: Vec<f64> = x.iter().zip(y).filter_map(|(x, y)| (x.is_finite() && y.is_finite()).then_some((y - a - b * x).abs())).collect();
        abs_residuals.sort_by(f64::total_cmp);
        let scale = abs_residuals[abs_residuals.len() / 2].max(1e-12);
        for i in 0..x.len() {
            let r = y[i] - a - b * x[i];
            let z = r.abs() / (tuning * scale);
            w[i] = if z <= 1.0 { 1.0 } else { 1.0 / z };
        }
    }
    let residuals: Vec<f64> = x.iter().zip(y).filter_map(|(x,y)| (x.is_finite()&&y.is_finite()).then_some(y-a-b*x)).collect();
    let scale = residuals.iter().map(|r| r*r).sum::<f64>().sqrt() / (residuals.len() as f64).sqrt();
    Some(LineFit { intercept:a, slope:b, residual_scale:scale, n:residuals.len() })
}

#[cfg(test)]
mod tests { use super::*;
    #[test] fn robust_line_recovers_slope_with_outlier() {
        let x = [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,20.0];
        let y = [1.0,3.0,5.0,7.0,9.0,11.0,13.0,15.0,17.0,19.0,1000.0];
        let f = huber_line(&x, &y, 30, 0.5).unwrap();
        assert!((f.slope - 2.0).abs() < 0.3);
    }
    #[test] fn multivariate_fit_recovers_known_coefficients() {
        let x1 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let y: Vec<f64> = x1.iter().zip(&x2).map(|(a, b)| 1.0 + 2.0 * a - 3.0 * b).collect();
        let fit = huber_regression(&[x1, x2], &y, 5, 1.345).unwrap();
        assert!((fit.coefficients[0] - 1.0).abs() < 1e-7);
        assert!((fit.coefficients[1] - 2.0).abs() < 1e-7);
        assert!((fit.coefficients[2] + 3.0).abs() < 1e-7);
    }
}

// --- Graph-scan region enumeration (paper 3 support) ---

/// Connected components of the valid-cell graph on a row-major grid.
///
/// Four-neighbour (rook) by default; eight-neighbour (queen) when `eight`
/// is set. Only cells with `valid[i] == true` become vertices. Returns
/// components as sorted-ascending cell lists in order of first discovery
/// (row-major), so output is deterministic. `None` on shape mismatch or
/// empty grid.
pub fn connected_components(valid: &[bool], rows: usize, cols: usize, eight: bool) -> Option<Vec<Vec<usize>>> {
    if rows.checked_mul(cols)? != valid.len() || rows == 0 || cols == 0 {
        return None;
    }
    let mut seen = vec![false; valid.len()];
    let mut out = Vec::new();
    for start in 0..valid.len() {
        if !valid[start] || seen[start] {
            continue;
        }
        let mut stack = vec![start];
        seen[start] = true;
        let mut comp = Vec::new();
        while let Some(i) = stack.pop() {
            comp.push(i);
            let r = i / cols;
            let c = i % cols;
            for dr in -1i64..=1 {
                for dc in -1i64..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    if !eight && dr != 0 && dc != 0 {
                        continue;
                    }
                    let nr = r as i64 + dr;
                    let nc = c as i64 + dc;
                    if nr < 0 || nc < 0 || nr >= rows as i64 || nc >= cols as i64 {
                        continue;
                    }
                    let j = nr as usize * cols + nc as usize;
                    if valid[j] && !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        comp.sort_unstable();
        out.push(comp);
    }
    Some(out)
}

#[derive(Debug, Clone, PartialEq)]
pub struct RegionStat {
    pub cells: Vec<usize>,
    /// Finite-cell support (cells actually scored).
    pub area: usize,
    /// Signed scan score `sum / sqrt(area)` over finite cells.
    pub score: f64,
    pub centroid_row: f64,
    pub centroid_col: f64,
    /// Rook edges facing outside the region, a missing cell, or the grid edge.
    pub perimeter: usize,
    /// Sum of squared differences over internal rook edges (both ends finite, in region).
    pub roughness: f64,
}

/// Score and describe one candidate region.
///
/// `region` holds flat grid indices (as produced by
/// [`connected_components`] or tile enumeration). Non-finite values are
/// skipped, never imputed; `None` when no cell is scorable.
pub fn region_stats(values: &[f64], region: &[usize], rows: usize, cols: usize) -> Option<RegionStat> {
    if rows.checked_mul(cols)? != values.len() || region.is_empty() {
        return None;
    }
    let mut in_region = vec![false; values.len()];
    for &i in region {
        if i >= values.len() {
            return None;
        }
        in_region[i] = true;
    }
    // Score over the set, so a duplicated index can never double-count.
    let mut sum = 0.0;
    let mut area = 0usize;
    let mut sr = 0.0;
    let mut sc = 0.0;
    for (i, inside) in in_region.iter().enumerate() {
        if !inside {
            continue;
        }
        let v = values[i];
        if v.is_finite() {
            sum += v;
            area += 1;
            sr += (i / cols) as f64;
            sc += (i % cols) as f64;
        }
    }
    if area == 0 {
        return None;
    }
    let mut perimeter = 0usize;
    let mut roughness = 0.0;
    for (i, inside) in in_region.iter().enumerate() {
        if !inside || !values[i].is_finite() {
            continue;
        }
        let r = i / cols;
        let c = i % cols;
        for (nr, nc) in [(r.wrapping_sub(1), c), (r + 1, c), (r, c.wrapping_sub(1)), (r, c + 1)] {
            let peer = nr < rows && nc < cols && in_region[nr * cols + nc] && values[nr * cols + nc].is_finite();
            if peer {
                // Count each internal edge once (right and down only).
                if (nr == r && nc == c + 1) || (nc == c && nr == r + 1) {
                    let d = values[i] - values[nr * cols + nc];
                    roughness += d * d;
                }
            } else {
                perimeter += 1;
            }
        }
    }
    Some(RegionStat {
        cells: region.to_vec(),
        area,
        score: sum / (area as f64).sqrt(),
        centroid_row: sr / area as f64,
        centroid_col: sc / area as f64,
        perimeter,
        roughness,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tile {
    pub row: usize,
    pub col: usize,
    pub size: usize,
}

/// Non-overlapping square tiling at each requested size (paper 3 multiscale family).
///
/// Tiles tile the grid left-to-right, top-to-bottom; edge tiles are
/// truncated, never padded. Sizes are inputs fixed before ranking;
/// non-positive sizes are skipped. Deterministic order: size-major, then row-major.
pub fn multiscale_tiles(rows: usize, cols: usize, sizes: &[usize]) -> Vec<Tile> {
    let mut out = Vec::new();
    for &size in sizes {
        if size == 0 {
            continue;
        }
        let mut r = 0;
        while r < rows {
            let mut c = 0;
            while c < cols {
                out.push(Tile { row: r, col: c, size });
                c += size;
            }
            r += size;
        }
    }
    out
}

/// Flat indices of a tile (truncated at the grid edge).
pub fn tile_cells(tile: Tile, rows: usize, cols: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for r in tile.row..(tile.row + tile.size).min(rows) {
        for c in tile.col..(tile.col + tile.size).min(cols) {
            out.push(r * cols + c);
        }
    }
    out
}

// --- Conditional-explanation block evaluation (paper 4 support) ---

/// Split row indices into train/test by block id.
///
/// Every row whose block is in `held_out` goes to test; the rest to
/// train. `None` when a held-out id is out of range or either side is
/// empty (a held-out evaluation needs both a fit and a test set).
pub fn block_split(blocks: &[usize], n_blocks: usize, held_out: &[usize]) -> Option<(Vec<usize>, Vec<usize>)> {
    if blocks.is_empty() || n_blocks == 0 {
        return None;
    }
    let mut is_held = vec![false; n_blocks];
    for &b in held_out {
        *is_held.get_mut(b)? = true;
    }
    let mut train = Vec::new();
    let mut test = Vec::new();
    for (i, &b) in blocks.iter().enumerate() {
        if b >= n_blocks {
            return None;
        }
        if is_held[b] {
            test.push(i);
        } else {
            train.push(i);
        }
    }
    if train.is_empty() || test.is_empty() {
        return None;
    }
    Some((train, test))
}

/// Predict row `index` from an intercept-first coefficient vector.
pub fn predict_row(coefficients: &[f64], features: &[Vec<f64>], index: usize) -> Option<f64> {
    if coefficients.is_empty() || coefficients.len() != features.len() + 1 {
        return None;
    }
    let mut y = coefficients[0];
    for (j, column) in features.iter().enumerate() {
        y += coefficients[j + 1] * *column.get(index)?;
    }
    Some(y)
}

/// Mean / mean-squared error over pairwise-finite entries. `None` when
/// lengths differ or nothing is scorable.
pub fn held_out_errors(actual: &[f64], predicted: &[f64]) -> Option<(f64, f64)> {
    if actual.len() != predicted.len() {
        return None;
    }
    let mut ae = 0.0;
    let mut se = 0.0;
    let mut n = 0usize;
    for (a, p) in actual.iter().zip(predicted) {
        if a.is_finite() && p.is_finite() {
            ae += (a - p).abs();
            se += (a - p).powi(2);
            n += 1;
        }
    }
    if n == 0 {
        return None;
    }
    Some((ae / n as f64, se / n as f64))
}

/// Relative held-out improvement of a fuller model over a baseline:
/// `(base - full) / base`. Positive means the added group helped.
/// `None` for a non-positive or non-finite baseline error.
pub fn error_reduction(baseline: f64, full: f64) -> Option<f64> {
    if !(baseline > 0.0) || !full.is_finite() {
        return None;
    }
    Some((baseline - full) / baseline)
}

/// Pearson correlation over pairwise-finite entries. `None` when fewer
/// than two pairs survive or either column is constant.
pub fn column_correlation(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    let pairs: Vec<(f64, f64)> = a
        .iter()
        .zip(b)
        .filter_map(|(x, y)| (x.is_finite() && y.is_finite()).then_some((*x, *y)))
        .collect();
    if pairs.len() < 2 {
        return None;
    }
    let n = pairs.len() as f64;
    let mx = pairs.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (x, y) in pairs {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx).powi(2);
        syy += (y - my).powi(2);
    }
    if !(sxx > 0.0) || !(syy > 0.0) {
        return None;
    }
    Some(sxy / (sxx * syy).sqrt())
}

/// Flagged column pairs with `|r| >= threshold`, sorted by descending
/// magnitude. Collinearity is reported, never hidden by dropping columns.
pub fn collinear_flags(features: &[Vec<f64>], threshold: f64) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::new();
    if !threshold.is_finite() {
        return out;
    }
    for i in 0..features.len() {
        for j in (i + 1)..features.len() {
            if let Some(r) = column_correlation(&features[i], &features[j]) {
                if r.abs() >= threshold {
                    out.push((i, j, r));
                }
            }
        }
    }
    out.sort_by(|a, b| b.2.abs().total_cmp(&a.2.abs()));
    out
}

/// Row indices with finite response and finite entries in every feature
/// column. Covariates are never imputed; incomplete rows are excluded and
/// counted by the caller (`n_total - complete.len()` missing).
pub fn complete_rows(y: &[f64], features: &[Vec<f64>]) -> Vec<usize> {
    (0..y.len())
        .filter(|&i| y[i].is_finite() && features.iter().all(|c| c.get(i).is_some_and(|v| v.is_finite())))
        .collect()
}
