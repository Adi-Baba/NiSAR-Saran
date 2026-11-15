//! Complete paired GCOV audit runner.
//!
//! The runner catalogs every common HDF5 path, writes a checkpoint after every
//! path, and emits numeric, complex, spatial, robust-model, permutation, MaxT,
//! FDR, evidence-ledger, figure, and checksum artifacts.

use audit_complex::{circular_resultant, magnitude, paired_ratio, phase, Complex};
use audit_core::summary;
use audit_distribution::{energy_distance, gaussian_mmd_squared};
use audit_hdf5::{inventory_pair, stream_complex_f32, stream_numeric_f64};
use audit_inference::{benjamini_hochberg_adjusted, max_statistic, sign_flip_mean_p};
use audit_ledger::{classify, evidence_tsv, EvidenceRow};
use audit_model::{huber_line, huber_regression};
use audit_spatial::{laplacian_energy, max_fixed_window_scan};
use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
};

const SAMPLE_CAP: usize = 200_000;
const DISTRIBUTION_CAP: usize = 512;
const REGRESSION_CAP: usize = 512;
const INFERENCE_CAP: usize = 2_048;
const TILE_ROWS: usize = 8_192;
const SIGN_PERMUTATIONS: usize = 999;
const MAXT_PERMUTATIONS: usize = 199;
/// Seed scheme for the per-path sign-flip nulls: FNV-1a over the dataset
/// path bytes (see `path_seed`). The seed is a pure function of the path,
/// so it is reproducible without storage and is recorded in the manifest.
pub const SIGN_SEED_SCHEME: &str = "fnv1a64(path_bytes)";
/// Seed for the dataset-level MaxT null stream. One xorshift64 stream is
/// shared across eligible rows within each replicate in row order; the seed
/// and the stream order are both fixed, so the null distribution replays
/// bit-identically from the manifest alone.
pub const MAXT_SEED: u64 = 0x91e10da5c79e7b1d;

#[derive(Debug, Clone)]
struct AuditRow {
    path: String,
    dtype: String,
    shape: String,
    status: String,
    sample_n: usize,
    median_delta: Option<f64>,
    mad_delta: Option<f64>,
    q05: Option<f64>,
    q95: Option<f64>,
    positive_fraction: Option<f64>,
    robust_effect: Option<f64>,
    energy_distance: Option<f64>,
    mmd_squared: Option<f64>,
    laplacian_energy: Option<f64>,
    window_score: Option<f64>,
    window_row: Option<usize>,
    window_col: Option<usize>,
    robust_slope: Option<f64>,
    robust_residual_scale: Option<f64>,
    phase_mean: Option<f64>,
    phase_concentration: Option<f64>,
    magnitude_ratio_median: Option<f64>,
    p_raw: Option<f64>,
    p_adjusted: Option<f64>,
    p_maxt: Option<f64>,
    quality_overlap: Option<f64>,
    classification: String,
    note: String,
    test_values: Vec<f64>,
}

impl AuditRow {
    fn unsupported(path: &str, dtype: &str, shape: &[usize], note: &str) -> Self {
        Self {
            path: path.into(),
            dtype: dtype.into(),
            shape: format!("{shape:?}"),
            status: "non_numeric_or_unsupported".into(),
            sample_n: 0,
            median_delta: None,
            mad_delta: None,
            q05: None,
            q95: None,
            positive_fraction: None,
            robust_effect: None,
            energy_distance: None,
            mmd_squared: None,
            laplacian_energy: None,
            window_score: None,
            window_row: None,
            window_col: None,
            robust_slope: None,
            robust_residual_scale: None,
            phase_mean: None,
            phase_concentration: None,
            magnitude_ratio_median: None,
            p_raw: None,
            p_adjusted: None,
            p_maxt: None,
            quality_overlap: None,
            classification: "uninformative_or_unsupported".into(),
            note: note.into(),
            test_values: Vec::new(),
        }
    }
}

fn stride_for(shape: &[usize]) -> usize {
    if shape.len() != 2 {
        return 1;
    }
    let elements = shape[0].saturating_mul(shape[1]);
    if elements <= SAMPLE_CAP {
        return 1;
    }
    let mut stride = ((elements as f64 / SAMPLE_CAP as f64).sqrt().ceil() as usize).max(1);
    while shape[0]
        .div_ceil(stride)
        .saturating_mul(shape[1].div_ceil(stride))
        > SAMPLE_CAP
    {
        stride += 1;
    }
    stride
}

fn thin(values: &[f64], cap: usize) -> Vec<f64> {
    if values.len() <= cap {
        return values.to_vec();
    }
    if cap <= 1 {
        return vec![values[0]];
    }
    (0..cap)
        .map(|i| values[i * (values.len() - 1) / (cap - 1)])
        .collect()
}

fn thin_pair(before: &[f64], after: &[f64], cap: usize) -> (Vec<f64>, Vec<f64>) {
    let n = before.len().min(after.len());
    if n <= cap {
        return (before[..n].to_vec(), after[..n].to_vec());
    }
    let indices: Vec<usize> = (0..cap).map(|i| i * (n - 1) / (cap - 1)).collect();
    (
        indices.iter().map(|i| before[*i]).collect(),
        indices.iter().map(|i| after[*i]).collect(),
    )
}

fn sample_numeric(path: &Path, dataset: &str, shape: &[usize]) -> Result<Option<Vec<f64>>, String> {
    let stride = stride_for(shape);
    let mut out = Vec::with_capacity(SAMPLE_CAP.min(16_384));
    let ok = stream_numeric_f64(path, dataset, TILE_ROWS, stride, |tile| {
        if tile.len() > SAMPLE_CAP && out.is_empty() {
            out = thin(tile, SAMPLE_CAP);
        } else if out.len() < SAMPLE_CAP {
            let remaining = SAMPLE_CAP - out.len();
            out.extend(tile.iter().take(remaining).copied());
        }
    })?;
    Ok(ok.then_some(out))
}

fn sample_complex(
    path: &Path,
    dataset: &str,
    shape: &[usize],
) -> Result<Option<Vec<Complex>>, String> {
    let stride = stride_for(shape);
    let mut out = Vec::with_capacity(SAMPLE_CAP.min(16_384));
    let ok = stream_complex_f32(path, dataset, TILE_ROWS, stride, |tile| {
        if out.len() < SAMPLE_CAP {
            let remaining = SAMPLE_CAP - out.len();
            out.extend(
                tile.iter()
                    .take(remaining)
                    .map(|z| Complex { re: z.re, im: z.im }),
            );
        }
    })?;
    Ok(ok.then_some(out))
}

fn sampled_grid(shape: &[usize], stride: usize, n: usize) -> Option<(usize, usize)> {
    if shape.len() != 2 {
        return None;
    }
    let rows = shape[0].div_ceil(stride);
    let cols = shape[1].div_ceil(stride);
    (rows.checked_mul(cols) == Some(n)).then_some((rows, cols))
}

fn path_seed(path: &str) -> u64 {
    path.bytes().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ byte as u64).wrapping_mul(0x100000001b3)
    })
}

fn aligned_delta(before: &[f64], after: &[f64]) -> Vec<f64> {
    before
        .iter()
        .zip(after)
        .map(|(b, a)| {
            if b.is_finite() && a.is_finite() {
                a - b
            } else {
                f64::NAN
            }
        })
        .collect()
}

#[derive(Debug, Clone)]
struct CalibrationReportRow {
    dataset_path: String,
    scale_path: String,
    before_scale: Option<f64>,
    after_scale: Option<f64>,
    power_offset_db: Option<f64>,
    raw_mean_db: Option<f64>,
    corrected_mean_db: Option<f64>,
    raw_median_db: Option<f64>,
    corrected_median_db: Option<f64>,
    raw_robust_effect: Option<f64>,
    corrected_robust_effect: Option<f64>,
    sampled_n: usize,
    valid_n: usize,
    mask_excluded: usize,
    exception_excluded: usize,
    quality_status: String,
    status: String,
}

fn calibration_report_tsv(rows: &[CalibrationReportRow]) -> String {
    let mut text = String::from("dataset_path\tscale_path\tbefore_scale_factor\tafter_scale_factor\tpower_offset_db\traw_mean_delta_db\tcorrected_mean_delta_db\traw_median_delta_db\tcorrected_median_delta_db\traw_robust_effect\tcorrected_robust_effect\tsampled_n\tvalid_n\tmask_excluded\texception_excluded\tquality_status\tstatus\n");
    for row in rows {
        text.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            row.dataset_path,
            row.scale_path,
            f(row.before_scale),
            f(row.after_scale),
            f(row.power_offset_db),
            f(row.raw_mean_db),
            f(row.corrected_mean_db),
            f(row.raw_median_db),
            f(row.corrected_median_db),
            f(row.raw_robust_effect),
            f(row.corrected_robust_effect),
            row.sampled_n,
            row.valid_n,
            row.mask_excluded,
            row.exception_excluded,
            row.quality_status,
            row.status,
        ));
    }
    text
}

fn channel_calibration_path(path: &str) -> Option<String> {
    let (frequency, polarization) = if path.ends_with("/HHHH") {
        ("frequencyA", "HH")
    } else if path.ends_with("/HVHV") {
        ("frequencyA", "HV")
    } else if path.ends_with("/frequencyB/HHHH") {
        ("frequencyB", "HH")
    } else if path.ends_with("/frequencyB/HVHV") {
        ("frequencyB", "HV")
    } else {
        return None;
    };
    // The first two branches need their frequency from the dataset path.
    let frequency = if path.contains("/frequencyB/") {
        "frequencyB"
    } else {
        frequency
    };
    Some(format!(
        "/science/LSAR/GCOV/metadata/calibrationInformation/{frequency}/{polarization}/scaleFactor"
    ))
}

fn scalar_f64(path: &Path, dataset: &str) -> Result<Option<f64>, String> {
    let mut values = Vec::new();
    let supported = stream_numeric_f64(path, dataset, 1, 1, |tile| values.extend_from_slice(tile))?;
    if !supported {
        return Ok(None);
    }
    Ok(values.into_iter().find(|value| value.is_finite()))
}

fn power_scale_offset_db(before: Option<f64>, after: Option<f64>) -> Option<f64> {
    match (before, after) {
        (Some(b), Some(a)) if b.is_finite() && a.is_finite() && b > 0.0 && a > 0.0 => {
            Some(20.0 * (a / b).log10())
        }
        _ => None,
    }
}

fn quality_path(path: &str, field: &str) -> Option<String> {
    let frequency = if path.contains("/frequencyB/") {
        "frequencyB"
    } else if path.contains("/frequencyA/") {
        "frequencyA"
    } else {
        return None;
    };
    path.split("/grids/")
        .next()
        .map(|prefix| format!("{prefix}/grids/{frequency}/{field}"))
}

fn finite_mean(values: &[f64]) -> Option<f64> {
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    (!finite.is_empty()).then(|| finite.iter().sum::<f64>() / finite.len() as f64)
}

fn apply_quality_gate(
    path: &str,
    shape: &[usize],
    before: &[f64],
    after: &[f64],
    quality_samples: &HashMap<String, (Vec<f64>, Vec<f64>)>,
) -> (Vec<f64>, Vec<f64>, usize, usize, String) {
    let Some(mask_path) = quality_path(path, "mask") else {
        return (
            before.to_vec(),
            after.to_vec(),
            0,
            0,
            "not_a_backscatter_grid".into(),
        );
    };
    let exception_path = quality_path(path, "inputDataExceptionMask");
    let mask = quality_samples.get(&mask_path);
    let exception = exception_path.as_ref().and_then(|p| quality_samples.get(p));
    let mask_aligned = mask
        .filter(|(b, a)| b.len() == before.len() && a.len() == after.len() && b.len() == a.len());
    let exception_aligned = exception
        .filter(|(b, a)| b.len() == before.len() && a.len() == after.len() && b.len() == a.len());
    if mask_aligned.is_none() {
        return (
            before.to_vec(),
            after.to_vec(),
            0,
            0,
            format!("mask_unavailable_or_shape_mismatch_{shape:?}"),
        );
    }
    let (mask_before, mask_after) = mask_aligned.unwrap();
    let mut gated_before = before.to_vec();
    let mut gated_after = after.to_vec();
    let mut mask_excluded = 0usize;
    let mut exception_excluded = 0usize;
    for i in 0..before.len() {
        let bad_mask = mask_before[i] != 1.0 || mask_after[i] != 1.0;
        let bad_exception = exception_aligned
            .map(|(b, a)| b[i] != 0.0 || a[i] != 0.0)
            .unwrap_or(false);
        if bad_mask {
            mask_excluded += 1;
        }
        if bad_exception {
            exception_excluded += 1;
        }
        if bad_mask || bad_exception {
            gated_before[i] = f64::NAN;
            gated_after[i] = f64::NAN;
        }
    }
    let status = if exception_aligned.is_some() {
        "mask_eq_1_and_input_exception_eq_0"
    } else {
        "mask_eq_1;input_exception_shape_mismatch_not_applied"
    };
    (
        gated_before,
        gated_after,
        mask_excluded,
        exception_excluded,
        status.into(),
    )
}

fn transform_analysis_units(
    path: &str,
    mut before: Vec<f64>,
    mut after: Vec<f64>,
) -> (Vec<f64>, Vec<f64>) {
    if path.ends_with("/HHHH") || path.ends_with("/HVHV") {
        for value in before.iter_mut().chain(after.iter_mut()) {
            *value = if value.is_finite() && *value > 0.0 {
                10.0 * value.log10()
            } else {
                f64::NAN
            };
        }
    }
    (before, after)
}

fn numeric_row(
    path: &str,
    dtype: &str,
    shape: &[usize],
    before: &[f64],
    after: &[f64],
) -> AuditRow {
    if before.len() != after.len() || before.is_empty() {
        return AuditRow::unsupported(path, dtype, shape, "unequal_or_empty_numeric_sample");
    }
    let delta = aligned_delta(before, after);
    let s = summary(&delta);
    let dist_before = thin(before, DISTRIBUTION_CAP);
    let dist_after = thin(after, DISTRIBUTION_CAP);
    let ed = energy_distance(&dist_before, &dist_after);
    let bandwidth = 1.4826 * s.mad.unwrap_or(1.0).max(1e-12);
    let mmd = gaussian_mmd_squared(&dist_before, &dist_after, bandwidth);
    let stride = stride_for(shape);
    let grid = sampled_grid(shape, stride, delta.len());
    let laplacian = grid.and_then(|(rows, cols)| laplacian_energy(&delta, rows, cols));
    let scan = grid.and_then(|(rows, cols)| {
        let wr = rows.min(4);
        let wc = cols.min(4);
        max_fixed_window_scan(&delta, rows, cols, wr, wc)
    });
    let (regression_before, regression_after) = thin_pair(before, after, REGRESSION_CAP);
    let line = huber_line(&regression_before, &regression_after, 5, 1.345);
    let test_values = thin(&delta, INFERENCE_CAP);
    let p_raw = sign_flip_mean_p(&test_values, SIGN_PERMUTATIONS, path_seed(path));
    AuditRow {
        path: path.into(),
        dtype: dtype.into(),
        shape: format!("{shape:?}"),
        status: "numeric_sampled".into(),
        sample_n: s.finite,
        median_delta: s.median,
        mad_delta: s.mad,
        q05: s.q05,
        q95: s.q95,
        positive_fraction: s.positive_fraction,
        robust_effect: s.robust_effect,
        energy_distance: ed,
        mmd_squared: mmd,
        laplacian_energy: laplacian,
        window_score: scan.map(|x| x.score),
        window_row: scan.map(|x| x.row),
        window_col: scan.map(|x| x.col),
        robust_slope: line.map(|x| x.slope),
        robust_residual_scale: line.map(|x| x.residual_scale),
        phase_mean: None,
        phase_concentration: None,
        magnitude_ratio_median: None,
        p_raw,
        p_adjusted: None,
        p_maxt: None,
        quality_overlap: None,
        classification: "pending_multiplicity".into(),
        note: format!(
            "adaptive_systematic_sample_stride_{stride}_cap_{SAMPLE_CAP}_transform_{}",
            if path.ends_with("/HHHH") || path.ends_with("/HVHV") {
                "power_to_db"
            } else {
                "identity"
            }
        ),
        test_values,
    }
}

fn complex_row(
    path: &str,
    dtype: &str,
    shape: &[usize],
    before: &[Complex],
    after: &[Complex],
) -> AuditRow {
    if before.len() != after.len() || before.is_empty() {
        return AuditRow::unsupported(path, dtype, shape, "unequal_or_empty_complex_sample");
    }
    let mut phases = Vec::new();
    let mut magnitudes = Vec::new();
    for (b, a) in before.iter().zip(after) {
        if ![b.re, b.im, a.re, a.im].iter().all(|x| x.is_finite()) {
            continue;
        }
        if let Some(ratio) = paired_ratio(*a, *b) {
            phases.push(phase(ratio));
            magnitudes.push(magnitude(ratio));
        }
    }
    let magnitude_summary = summary(&magnitudes);
    let phase_summary = summary(&phases);
    let circular = circular_resultant(&phases);
    let test_values = thin(&phases, INFERENCE_CAP);
    let p_raw = sign_flip_mean_p(&test_values, SIGN_PERMUTATIONS, path_seed(path));
    AuditRow {
        path: path.into(),
        dtype: dtype.into(),
        shape: format!("{shape:?}"),
        status: "complex_f32_sampled".into(),
        sample_n: phases.len(),
        median_delta: phase_summary.median,
        mad_delta: phase_summary.mad,
        q05: phase_summary.q05,
        q95: phase_summary.q95,
        positive_fraction: phase_summary.positive_fraction,
        robust_effect: phase_summary.robust_effect,
        energy_distance: None,
        mmd_squared: None,
        laplacian_energy: None,
        window_score: None,
        window_row: None,
        window_col: None,
        robust_slope: None,
        robust_residual_scale: None,
        phase_mean: circular.map(|x| x.0),
        phase_concentration: circular.map(|x| x.1),
        magnitude_ratio_median: magnitude_summary.median,
        p_raw,
        p_adjusted: None,
        p_maxt: None,
        quality_overlap: None,
        classification: "pending_multiplicity".into(),
        note: format!("typed_complex_f32_adaptive_stride_{}", stride_for(shape)),
        test_values,
    }
}

fn f(value: Option<f64>) -> String {
    value
        .map(|x| format!("{x:.12e}"))
        .unwrap_or_else(|| "NA".into())
}
fn u(value: Option<usize>) -> String {
    value.map(|x| x.to_string()).unwrap_or_else(|| "NA".into())
}

fn dataset_tsv(rows: &[AuditRow]) -> String {
    let mut text = String::from("path\tdtype\tshape\tstatus\tsample_n\tmedian_delta\tmad_delta\tq05\tq95\tpositive_fraction\trobust_effect\tenergy_distance\tmmd_squared\tlaplacian_energy\twindow_score\twindow_row\twindow_col\trobust_slope\trobust_residual_scale\tphase_mean\tphase_concentration\tmagnitude_ratio_median\tp_raw\tp_adjusted_bh\tp_maxt\tquality_overlap\tclassification\tnote\n");
    for row in rows {
        text.push_str(&format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            row.path, row.dtype.replace('\t', " "), row.shape, row.status, row.sample_n,
            f(row.median_delta), f(row.mad_delta), f(row.q05), f(row.q95),
            f(row.positive_fraction), f(row.robust_effect), f(row.energy_distance),
            f(row.mmd_squared), f(row.laplacian_energy), f(row.window_score),
            u(row.window_row), u(row.window_col), f(row.robust_slope),
            f(row.robust_residual_scale), f(row.phase_mean), f(row.phase_concentration),
            f(row.magnitude_ratio_median), f(row.p_raw), f(row.p_adjusted),
            f(row.p_maxt), f(row.quality_overlap), row.classification, row.note));
    }
    text
}

fn apply_multiplicity(rows: &mut [AuditRow]) {
    let tested: Vec<usize> = rows
        .iter()
        .enumerate()
        .filter_map(|(i, row)| row.p_raw.map(|_| i))
        .collect();
    let p_values: Vec<f64> = tested.iter().map(|i| rows[*i].p_raw.unwrap()).collect();
    for (index, adjusted) in tested.iter().zip(benjamini_hochberg_adjusted(&p_values)) {
        rows[*index].p_adjusted = Some(adjusted);
    }
    let mut state = MAXT_SEED;
    let mut null_maxima = Vec::with_capacity(MAXT_PERMUTATIONS);
    for _ in 0..MAXT_PERMUTATIONS {
        let mut maximum: f64 = 0.0;
        for row in rows.iter().filter(|row| maxt_eligible(row)) {
            let scale = (row
                .test_values
                .iter()
                .filter(|x| x.is_finite())
                .map(|x| x * x)
                .sum::<f64>()
                / row
                    .test_values
                    .iter()
                    .filter(|x| x.is_finite())
                    .count()
                    .max(1) as f64)
                .sqrt()
                .max(1e-12);
            let mut sum = 0.0;
            let mut n = 0usize;
            for value in &row.test_values {
                if !value.is_finite() {
                    continue;
                }
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                sum += if state & 1 == 0 {
                    *value / scale
                } else {
                    -*value / scale
                };
                n += 1;
            }
            if n > 0 {
                maximum = maximum.max((sum / n as f64).abs());
            }
        }
        null_maxima.push(maximum);
    }
    for row in rows.iter_mut().filter(|row| maxt_eligible(row)) {
        let finite: Vec<f64> = row
            .test_values
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        if finite.is_empty() {
            continue;
        }
        let scale = (finite.iter().map(|x| x * x).sum::<f64>() / finite.len() as f64)
            .sqrt()
            .max(1e-12);
        let observed = (finite.iter().sum::<f64>() / finite.len() as f64 / scale).abs();
        row.p_maxt = max_statistic(observed, &null_maxima);
    }
}

fn maxt_eligible(row: &AuditRow) -> bool {
    row.sample_n >= 30
        && row.path.contains("/grids/")
        && [
            "/HHHH",
            "/HVHV",
            "/mask",
            "/inputDataExceptionMask",
            "/numberOfLooks",
            "/rtcGammaToSigmaFactor",
        ]
        .iter()
        .any(|suffix| row.path.ends_with(suffix))
        && !row.test_values.is_empty()
}

fn candidate_priority(row: &AuditRow) -> u8 {
    match row.classification.as_str() {
        "unknown_reproducible_anomaly" => 0,
        "quality_geometry_consistent" => 1,
        "instrument_processing_consistent_calibration_field_change" => 2,
        "uninformative" | "uninformative_quality_geometry" => 3,
        _ => 4,
    }
}

fn quality_overlap(signal: &[f64], quality_fields: &[Vec<f64>]) -> Option<f64> {
    if signal.is_empty()
        || quality_fields.is_empty()
        || quality_fields.iter().any(|x| x.len() != signal.len())
    {
        return None;
    }
    let abs: Vec<f64> = signal
        .iter()
        .filter(|x| x.is_finite())
        .map(|x| x.abs())
        .collect();
    let threshold = audit_core::quantile(&abs, 0.95)?;
    let quality_thresholds: Vec<f64> = quality_fields
        .iter()
        .map(|field| {
            let finite_abs: Vec<f64> = field
                .iter()
                .filter(|x| x.is_finite())
                .map(|x| x.abs())
                .collect();
            audit_core::quantile(&finite_abs, 0.95).unwrap_or(f64::INFINITY)
        })
        .collect();
    let mut candidates = 0usize;
    let mut overlaps = 0usize;
    for i in 0..signal.len() {
        if !signal[i].is_finite() || signal[i].abs() < threshold {
            continue;
        }
        candidates += 1;
        if quality_fields
            .iter()
            .zip(&quality_thresholds)
            .any(|(field, q)| {
                field[i].is_finite()
                    && if *q > 0.0 {
                        field[i].abs() >= *q
                    } else {
                        field[i].abs() > 0.0
                    }
            })
        {
            overlaps += 1;
        }
    }
    (candidates > 0).then_some(overlaps as f64 / candidates as f64)
}

fn final_classification(row: &AuditRow) -> String {
    let supported = row.status == "numeric_sampled" || row.status == "complex_f32_sampled";
    if !supported {
        return "uninformative_or_unsupported".into();
    }
    if row.path.contains("/identification/") {
        return "uninformative_expected_identification_change".into();
    }
    if row.path.contains("/metadata/") {
        if row.status == "complex_f32_sampled"
            && row.p_adjusted.map(|p| p <= 0.05).unwrap_or(false)
            && row.robust_effect.map(|e| e.abs() >= 0.5).unwrap_or(false)
        {
            return "instrument_processing_consistent_calibration_field_change".into();
        }
        return "uninformative_expected_processing_metadata_change".into();
    }
    if row.path.ends_with("/mask")
        || row.path.ends_with("/inputDataExceptionMask")
        || row.path.ends_with("/numberOfLooks")
        || row.path.ends_with("/rtcGammaToSigmaFactor")
    {
        return if row.p_adjusted.map(|p| p <= 0.05).unwrap_or(false)
            && row.robust_effect.map(|e| e.abs() >= 0.5).unwrap_or(false)
        {
            "quality_geometry_consistent"
        } else {
            "uninformative_quality_geometry"
        }
        .into();
    }
    if row.path.ends_with("/xCoordinates")
        || row.path.ends_with("/yCoordinates")
        || row.path.ends_with("/projection")
        || row.path.ends_with("/xCoordinateSpacing")
        || row.path.ends_with("/yCoordinateSpacing")
    {
        return "uninformative_grid_coordinate_metadata".into();
    }
    classify(row.p_adjusted, row.robust_effect, row.quality_overlap, true).into()
}

fn ranked_svg(rows: &[AuditRow]) -> String {
    let mut ranked: Vec<&AuditRow> = rows
        .iter()
        .filter(|r| {
            r.path.contains("/grids/") && r.robust_effect.map(|x| x.is_finite()).unwrap_or(false)
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.robust_effect
            .unwrap()
            .abs()
            .total_cmp(&a.robust_effect.unwrap().abs())
    });
    ranked.truncate(20);
    let height = 80 + ranked.len() * 28;
    let mut svg = format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1200\" height=\"{height}\" viewBox=\"0 0 1200 {height}\"><rect width=\"100%\" height=\"100%\" fill=\"white\"/><text x=\"20\" y=\"30\" font-family=\"sans-serif\" font-size=\"20\">Top absolute robust paired effects (screening)</text>");
    let maximum = ranked
        .iter()
        .filter_map(|r| r.robust_effect)
        .map(f64::abs)
        .fold(0.0, f64::max)
        .max(1e-12);
    for (i, row) in ranked.iter().enumerate() {
        let y = 55 + i * 28;
        let effect = row.robust_effect.unwrap();
        let width = (effect.abs() / maximum * 360.0).min(360.0);
        let label = row
            .path
            .rsplit('/')
            .next()
            .unwrap_or(&row.path)
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;");
        svg.push_str(&format!("<text x=\"20\" y=\"{}\" font-family=\"monospace\" font-size=\"13\">{}</text><rect x=\"420\" y=\"{}\" width=\"{:.2}\" height=\"16\" fill=\"{}\"/><text x=\"790\" y=\"{}\" font-family=\"monospace\" font-size=\"12\">{:.4}</text>", y + 13, label, y, width, if effect >= 0.0 { "#2166ac" } else { "#b2182b" }, y + 13, effect));
    }
    svg.push_str("</svg>");
    svg
}

/// The manifest parameters line. Any change here changes what the runner
/// emits, so the test below pins it against the committed October value.
fn manifest_parameters_line(n_paths: usize) -> String {
    format!("parameters\t0\tmatching_paths={n_paths};sample_cap={SAMPLE_CAP};distribution_cap={DISTRIBUTION_CAP};regression_cap={REGRESSION_CAP};sign_permutations={SIGN_PERMUTATIONS};maxt_permutations={MAXT_PERMUTATIONS};sign_seed={SIGN_SEED_SCHEME};maxt_seed=0x{MAXT_SEED:016x}")
}

fn manifest_calibration_line() -> &'static str {
    "calibration\t0\tmode=metadata_scale_factor_power_domain;power_offset=20*log10(after_scale_factor/before_scale_factor);quality_gate=mask_eq_1_and_inputDataExceptionMask_eq_0_when_shapes_align;raw_audit=uncalibrated_dataset_audit.tsv"
}

fn hash_line(path: &Path) -> Result<String, String> {
    let (bytes, hash) =
        nz_core::sha256_hex_file(path).map_err(|e| format!("hash {}: {e:?}", path.display()))?;
    Ok(format!("{}\t{}\t{}", path.display(), bytes, hash))
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        return Err(format!("usage: {} BEFORE.h5 AFTER.h5 OUTPUT_DIR", args[0]));
    }
    let before_path = PathBuf::from(&args[1]);
    let after_path = PathBuf::from(&args[2]);
    let output_dir = PathBuf::from(&args[3]);
    fs::create_dir_all(&output_dir).map_err(|e| e.to_string())?;
    let checkpoint = output_dir.join("dataset_audit.tsv");
    let raw_checkpoint = output_dir.join("uncalibrated_dataset_audit.tsv");
    let inventory = inventory_pair(&before_path, &after_path)?;
    let mut rows = Vec::with_capacity(inventory.matching_paths.len());
    let mut raw_rows = Vec::with_capacity(inventory.matching_paths.len());
    let mut retained: HashMap<String, (Vec<f64>, Vec<f64>)> = HashMap::new();

    // Read each quality field once.  Quality fields are only used when their
    // sampled shape aligns exactly with the science field; frequency-B's
    // exception mask is deliberately not forced onto its smaller raster.
    let mut quality_samples: HashMap<String, (Vec<f64>, Vec<f64>)> = HashMap::new();
    for frequency in ["frequencyA", "frequencyB"] {
        for field in ["mask", "inputDataExceptionMask"] {
            let path = format!("/science/LSAR/GCOV/grids/{frequency}/{field}");
            let Some(entry) = inventory
                .before
                .datasets
                .iter()
                .find(|entry| entry.path == path)
            else {
                continue;
            };
            let Some(before) = sample_numeric(&before_path, &path, &entry.shape)? else {
                continue;
            };
            let Some(after) = sample_numeric(&after_path, &path, &entry.shape)? else {
                continue;
            };
            quality_samples.insert(path, (before, after));
        }
    }

    let mut calibration_scales: HashMap<String, (Option<f64>, Option<f64>)> = HashMap::new();
    for path in inventory
        .matching_paths
        .iter()
        .filter_map(|path| channel_calibration_path(path))
    {
        let before = scalar_f64(&before_path, &path).unwrap_or(None);
        let after = scalar_f64(&after_path, &path).unwrap_or(None);
        calibration_scales.insert(path, (before, after));
    }
    let mut calibration_rows = Vec::new();

    for (index, path) in inventory.matching_paths.iter().enumerate() {
        eprintln!(
            "[{}/{}] {}",
            index + 1,
            inventory.matching_paths.len(),
            path
        );
        let entry = inventory
            .before
            .datasets
            .iter()
            .find(|entry| &entry.path == path)
            .ok_or_else(|| format!("missing catalog entry {path}"))?;
        let before_numeric = sample_numeric(&before_path, path, &entry.shape)?;
        let after_numeric = sample_numeric(&after_path, path, &entry.shape)?;
        let (raw_row, row) = match (before_numeric, after_numeric) {
            (Some(before), Some(after)) => {
                let (before, after) = transform_analysis_units(path, before, after);
                let raw_row = numeric_row(path, &entry.dtype, &entry.shape, &before, &after);
                if let Some(scale_path) = channel_calibration_path(path) {
                    let (before_scale, after_scale) = calibration_scales
                        .get(&scale_path)
                        .copied()
                        .unwrap_or((None, None));
                    let power_offset_db = power_scale_offset_db(before_scale, after_scale);
                    let (
                        calibrated_before,
                        mut calibrated_after,
                        mask_excluded,
                        exception_excluded,
                        quality_status,
                    ) = apply_quality_gate(path, &entry.shape, &before, &after, &quality_samples);
                    if let Some(offset) = power_offset_db {
                        for value in &mut calibrated_after {
                            if value.is_finite() {
                                *value -= offset;
                            }
                        }
                    }
                    let mut calibrated_row = numeric_row(
                        path,
                        &entry.dtype,
                        &entry.shape,
                        &calibrated_before,
                        &calibrated_after,
                    );
                    calibrated_row.note = format!(
                        "{};scale_path={};power_offset_db={}",
                        calibrated_row.note,
                        scale_path,
                        f(power_offset_db),
                    );
                    if ["/HHHH", "/mask", "/numberOfLooks", "/rtcGammaToSigmaFactor"]
                        .iter()
                        .any(|suffix| path.ends_with(suffix))
                    {
                        retained.insert(
                            path.clone(),
                            (calibrated_before.clone(), calibrated_after.clone()),
                        );
                    }
                    let raw_delta = aligned_delta(&before, &after);
                    let corrected_delta = aligned_delta(&calibrated_before, &calibrated_after);
                    let raw_summary = summary(&raw_delta);
                    let corrected_summary = summary(&corrected_delta);
                    calibration_rows.push(CalibrationReportRow {
                        dataset_path: path.clone(),
                        scale_path,
                        before_scale,
                        after_scale,
                        power_offset_db,
                        raw_mean_db: finite_mean(&raw_delta),
                        corrected_mean_db: finite_mean(&corrected_delta),
                        raw_median_db: raw_summary.median,
                        corrected_median_db: corrected_summary.median,
                        raw_robust_effect: raw_summary.robust_effect,
                        corrected_robust_effect: corrected_summary.robust_effect,
                        sampled_n: before.len(),
                        valid_n: corrected_summary.finite,
                        mask_excluded,
                        exception_excluded,
                        quality_status: quality_status.clone(),
                        status: match (before_scale, after_scale, power_offset_db) {
                            (Some(_), Some(_), Some(offset)) if offset.abs() > 1e-12 => {
                                "scale_factor_applied"
                            }
                            (Some(_), Some(_), Some(_)) => "scale_factor_stable_no_offset",
                            _ => "no_valid_scale_factor_offset",
                        }
                        .into(),
                    });
                    (raw_row, calibrated_row)
                } else {
                    if ["/HHHH", "/mask", "/numberOfLooks", "/rtcGammaToSigmaFactor"]
                        .iter()
                        .any(|suffix| path.ends_with(suffix))
                    {
                        retained.insert(path.clone(), (before.clone(), after.clone()));
                    }
                    (raw_row.clone(), raw_row)
                }
            }
            (None, None) => {
                let before_complex = sample_complex(&before_path, path, &entry.shape)?;
                let after_complex = sample_complex(&after_path, path, &entry.shape)?;
                match (before_complex, after_complex) {
                    (Some(before), Some(after)) => {
                        let row = complex_row(path, &entry.dtype, &entry.shape, &before, &after);
                        (row.clone(), row)
                    }
                    _ => {
                        let row = AuditRow::unsupported(
                            path,
                            &entry.dtype,
                            &entry.shape,
                            "text_reference_or_unhandled_compound_recorded",
                        );
                        (row.clone(), row)
                    }
                }
            }
            _ => {
                let row = AuditRow::unsupported(
                    path,
                    &entry.dtype,
                    &entry.shape,
                    "before_after_type_support_mismatch",
                );
                (row.clone(), row)
            }
        };
        rows.push(row);
        raw_rows.push(raw_row);
        fs::write(&checkpoint, dataset_tsv(&rows)).map_err(|e| e.to_string())?;
        fs::write(&raw_checkpoint, dataset_tsv(&raw_rows)).map_err(|e| e.to_string())?;
    }

    let hhhh_path = "/science/LSAR/GCOV/grids/frequencyA/HHHH";
    let quality_paths = [
        "/science/LSAR/GCOV/grids/frequencyA/mask",
        "/science/LSAR/GCOV/grids/frequencyA/numberOfLooks",
        "/science/LSAR/GCOV/grids/frequencyA/rtcGammaToSigmaFactor",
    ];
    let mut conditional_text = String::from(
        "response\tpredictors\tn\tintercept\tcoefficients\tresidual_scale\tquality_overlap\n",
    );
    if let Some((before_h, after_h)) = retained.get(hhhh_path) {
        let signal = aligned_delta(before_h, after_h);
        let quality: Vec<Vec<f64>> = quality_paths
            .iter()
            .filter_map(|path| retained.get(*path).map(|(b, a)| aligned_delta(b, a)))
            .filter(|x| x.len() == signal.len())
            .collect();
        let overlap = quality_overlap(&signal, &quality);
        if let Some(row) = rows.iter_mut().find(|row| row.path == hhhh_path) {
            row.quality_overlap = overlap;
        }
        if !quality.is_empty() {
            if let Some(fit) = huber_regression(&quality, &signal, 10, 1.345) {
                conditional_text.push_str(&format!(
                    "{}\t{}\t{}\t{:.12e}\t{}\t{:.12e}\t{}\n",
                    hhhh_path,
                    quality_paths[..quality.len()].join("|"),
                    fit.n,
                    fit.coefficients[0],
                    fit.coefficients[1..]
                        .iter()
                        .map(|x| format!("{x:.12e}"))
                        .collect::<Vec<_>>()
                        .join("|"),
                    fit.residual_scale,
                    f(overlap)
                ));
            }
        }
    }

    apply_multiplicity(&mut raw_rows);
    for row in &mut raw_rows {
        row.classification = final_classification(row);
    }
    fs::write(&raw_checkpoint, dataset_tsv(&raw_rows)).map_err(|e| e.to_string())?;

    apply_multiplicity(&mut rows);
    for row in &mut rows {
        row.classification = final_classification(row);
    }
    fs::write(&checkpoint, dataset_tsv(&rows)).map_err(|e| e.to_string())?;
    let conditional_path = output_dir.join("conditional_hhhh.tsv");
    fs::write(&conditional_path, conditional_text).map_err(|e| e.to_string())?;
    let calibration_path = output_dir.join("calibration_report.tsv");
    fs::write(&calibration_path, calibration_report_tsv(&calibration_rows))
        .map_err(|e| e.to_string())?;

    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|a, b| {
        candidate_priority(&rows[*a])
            .cmp(&candidate_priority(&rows[*b]))
            .then_with(|| {
                rows[*a]
                    .p_adjusted
                    .unwrap_or(1.0)
                    .total_cmp(&rows[*b].p_adjusted.unwrap_or(1.0))
            })
            .then_with(|| {
                rows[*b]
                    .robust_effect
                    .unwrap_or(0.0)
                    .abs()
                    .total_cmp(&rows[*a].robust_effect.unwrap_or(0.0).abs())
            })
    });
    let ranked_path = output_dir.join("ranked_results.tsv");
    fs::write(
        &ranked_path,
        dataset_tsv(&order.iter().map(|i| rows[*i].clone()).collect::<Vec<_>>()),
    )
    .map_err(|e| e.to_string())?;
    let evidence: Vec<EvidenceRow> = order
        .iter()
        .enumerate()
        .map(|(rank, index)| {
            let row = &rows[*index];
            EvidenceRow {
                candidate_id: format!("C{:03}", rank + 1),
                dataset_path: row.path.clone(),
                family: if row.status.starts_with("complex") {
                    "complex_phase"
                } else {
                    "paired_dataset"
                }
                .into(),
                effect_size: row.robust_effect,
                spatial_support: row.window_score.map(f64::abs),
                quality_overlap: row.quality_overlap,
                p_raw: row.p_raw,
                p_adjusted: row.p_adjusted,
                classification: row.classification.clone(),
                competing_explanations: if row.quality_overlap.unwrap_or(0.0) >= 0.5 {
                    "quality_transition"
                } else {
                    "physical|environmental|unmeasured_processing"
                }
                .into(),
                reproducibility_status: "deterministic_rust_screen_complete".into(),
            }
        })
        .collect();
    let evidence_path = output_dir.join("evidence_ledger.tsv");
    fs::write(&evidence_path, evidence_tsv(&evidence)).map_err(|e| e.to_string())?;
    let figure_path = output_dir.join("ranked_effects.svg");
    fs::write(&figure_path, ranked_svg(&rows)).map_err(|e| e.to_string())?;

    let manifest_path = output_dir.join("run_manifest.tsv");
    let mut manifest = String::from("artifact\tbytes\tsha256\n");
    for path in [
        &before_path,
        &after_path,
        &checkpoint,
        &raw_checkpoint,
        &ranked_path,
        &evidence_path,
        &conditional_path,
        &calibration_path,
        &figure_path,
    ] {
        manifest.push_str(&hash_line(path)?);
        manifest.push('\n');
    }
    manifest.push_str(&manifest_parameters_line(rows.len()));
    manifest.push('\n');
    manifest.push_str(manifest_calibration_line());
    manifest.push('\n');
    fs::write(&manifest_path, manifest).map_err(|e| e.to_string())?;
    println!(
        "completed {} matching paths; artifacts: {}",
        rows.len(),
        output_dir.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adaptive_stride_never_clips_a_complete_grid() {
        let shape = [33_192, 33_840];
        let stride = stride_for(&shape);
        assert!(shape[0].div_ceil(stride) * shape[1].div_ceil(stride) <= SAMPLE_CAP);
        assert_eq!(stride, 76);
    }
    #[test]
    fn small_grids_are_never_strided() {
        assert_eq!(stride_for(&[333, 339]), 1);
        assert_eq!(stride_for(&[]), 1);
        assert_eq!(stride_for(&[4_149, 4_230]), 10);
    }
    #[test]
    fn null_seeds_are_named_and_recorded() {
        assert_eq!(MAXT_SEED, 0x91e10da5c79e7b1d);
        assert!(!SIGN_SEED_SCHEME.is_empty());
        assert_eq!(
            path_seed("/science/LSAR/GCOV/grids/frequencyA/HHHH"),
            path_seed("/science/LSAR/GCOV/grids/frequencyA/HHHH")
        );
        assert_ne!(
            path_seed("/science/LSAR/GCOV/grids/frequencyA/HHHH"),
            path_seed("/science/LSAR/GCOV/grids/frequencyB/HHHH")
        );
    }
    #[test]
    fn amplitude_scale_change_is_converted_to_power_db() {
        let offset = power_scale_offset_db(Some(1.0), Some(2.0)).unwrap();
        assert!((offset - 20.0 * 2.0_f64.log10()).abs() < 1e-12);
        assert_eq!(power_scale_offset_db(Some(0.0), Some(2.0)), None);
    }
    #[test]
    fn channel_calibration_paths_follow_frequency_and_polarization() {
        assert_eq!(
            channel_calibration_path("/science/LSAR/GCOV/grids/frequencyA/HHHH"),
            Some(
                "/science/LSAR/GCOV/metadata/calibrationInformation/frequencyA/HH/scaleFactor"
                    .into()
            )
        );
        assert_eq!(
            channel_calibration_path("/science/LSAR/GCOV/grids/frequencyB/HVHV"),
            Some(
                "/science/LSAR/GCOV/metadata/calibrationInformation/frequencyB/HV/scaleFactor"
                    .into()
            )
        );
        assert_eq!(
            channel_calibration_path("/science/LSAR/GCOV/grids/frequencyA/mask"),
            None
        );
    }
    #[test]
    fn quality_gate_requires_aligned_mask_and_exception_fields() {
        let mut fields = HashMap::new();
        fields.insert(
            "/science/LSAR/GCOV/grids/frequencyA/mask".into(),
            (vec![1.0, 0.0, 1.0], vec![1.0, 1.0, 1.0]),
        );
        fields.insert(
            "/science/LSAR/GCOV/grids/frequencyA/inputDataExceptionMask".into(),
            (vec![0.0, 0.0, 1.0], vec![0.0, 0.0, 0.0]),
        );
        let (before, after, mask_excluded, exception_excluded, status) = apply_quality_gate(
            "/science/LSAR/GCOV/grids/frequencyA/HHHH",
            &[2, 2],
            &[1.0, 2.0, 3.0],
            &[2.0, 3.0, 4.0],
            &fields,
        );
        assert_eq!(mask_excluded, 1);
        assert_eq!(exception_excluded, 1);
        assert!(before[1].is_nan() && after[1].is_nan());
        assert!(before[2].is_nan() && after[2].is_nan());
        assert_eq!(status, "mask_eq_1_and_input_exception_eq_0");
    }
    #[test]
    fn thinning_preserves_endpoints_and_cap() {
        let values: Vec<f64> = (0..10).map(|v| v as f64).collect();
        assert_eq!(thin(&values, 4), vec![0.0, 3.0, 6.0, 9.0]);
        assert_eq!(thin(&values, 64).len(), 10);
        let (before, after) = thin_pair(&values, &values, 3);
        assert_eq!((before, after), (vec![0.0, 4.0, 9.0], vec![0.0, 4.0, 9.0]));
    }
    fn eligible_row(path: &str, n: usize) -> AuditRow {
        let mut row = AuditRow::unsupported(path, "Float(U4)", &[100, 100], "test");
        row.status = "numeric_sampled".into();
        row.sample_n = n;
        row.test_values = vec![1.0; n.min(8)];
        row
    }
    #[test]
    fn maxt_family_is_the_prespecified_grid_set() {
        assert!(maxt_eligible(&eligible_row(
            "/science/LSAR/GCOV/grids/frequencyA/HHHH",
            64
        )));
        assert!(maxt_eligible(&eligible_row(
            "/science/LSAR/GCOV/grids/frequencyA/mask",
            64
        )));
        assert!(!maxt_eligible(&eligible_row(
            "/science/LSAR/GCOV/metadata/radarGrid/x",
            64
        )));
        assert!(!maxt_eligible(&eligible_row(
            "/science/LSAR/GCOV/grids/frequencyA/HHHH",
            29
        )));
        assert!(!maxt_eligible(&eligible_row(
            "/science/LSAR/GCOV/grids/frequencyA/HHHH",
            0
        )));
    }
    #[test]
    fn quality_overlap_needs_aligned_fields() {
        assert_eq!(quality_overlap(&[], &[vec![1.0]]), None);
        assert_eq!(quality_overlap(&[1.0, 2.0], &[vec![1.0]]), None);
        assert_eq!(
            quality_overlap(&vec![10.0; 100], &vec![vec![10.0; 100]]),
            Some(1.0)
        );
        assert_eq!(
            quality_overlap(&vec![10.0; 100], &vec![vec![0.0; 100]]),
            Some(0.0)
        );
    }
    fn classified_row(
        path: &str,
        p: Option<f64>,
        effect: Option<f64>,
        overlap: Option<f64>,
    ) -> AuditRow {
        let mut row = AuditRow::unsupported(path, "Float(U4)", &[8, 8], "test");
        row.status = "numeric_sampled".into();
        row.p_adjusted = p;
        row.robust_effect = effect;
        row.quality_overlap = overlap;
        row
    }
    #[test]
    fn final_classification_routes_every_family() {
        assert_eq!(
            final_classification(&AuditRow::unsupported("/x", "FixedAscii(4)", &[2], "t")),
            "uninformative_or_unsupported"
        );
        assert_eq!(
            final_classification(&classified_row(
                "/science/LSAR/GCOV/metadata/radarGrid/x",
                Some(0.001),
                Some(9.0),
                None
            )),
            "uninformative_expected_processing_metadata_change"
        );
        assert_eq!(
            final_classification(&classified_row(
                "/science/LSAR/GCOV/grids/frequencyA/mask",
                Some(0.001),
                Some(2.0),
                None
            )),
            "quality_geometry_consistent"
        );
        assert_eq!(
            final_classification(&classified_row(
                "/science/LSAR/GCOV/grids/frequencyA/mask",
                Some(0.9),
                Some(2.0),
                None
            )),
            "uninformative_quality_geometry"
        );
        assert_eq!(
            final_classification(&classified_row(
                "/science/LSAR/GCOV/grids/frequencyA/xCoordinates",
                None,
                None,
                None
            )),
            "uninformative_grid_coordinate_metadata"
        );
        assert_eq!(
            final_classification(&classified_row(
                "/science/LSAR/GCOV/grids/frequencyB/HHHH",
                Some(0.006),
                Some(1.45),
                Some(0.0)
            )),
            "unknown_reproducible_anomaly"
        );
    }
    #[test]
    fn dataset_tsv_renders_header_and_na() {
        let row = AuditRow::unsupported("/p", "FixedAscii(4)", &[2], "note");
        let text = dataset_tsv(std::slice::from_ref(&row));
        assert!(text.starts_with("path\tdtype\tshape\tstatus\tsample_n\tmedian_delta\t"));
        assert!(text.contains("/p\tFixedAscii(4)\t[2]\tnon_numeric_or_unsupported\t0\tNA\t"));
    }
    #[test]
    fn manifest_parameters_line_matches_the_october_artifact() {
        assert_eq!(manifest_parameters_line(288), "parameters\t0\tmatching_paths=288;sample_cap=200000;distribution_cap=512;regression_cap=512;sign_permutations=999;maxt_permutations=199;sign_seed=fnv1a64(path_bytes);maxt_seed=0x91e10da5c79e7b1d");
    }
    #[test]
    fn ranked_figure_filters_and_escapes() {
        let mut good = eligible_row("/science/LSAR/GCOV/grids/a&b<H>", 64);
        good.robust_effect = Some(1.5);
        let mut bad = eligible_row("/science/LSAR/GCOV/metadata/x", 64);
        bad.robust_effect = Some(99.0);
        let svg = ranked_svg(&[bad, good]);
        assert!(svg.contains("a&amp;b&lt;H&gt;"));
        assert!(!svg.contains("a&b<H>"));
        assert!(!svg.contains("99.0000"));
    }
}
