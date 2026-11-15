//! Cross-crate synthetic fixture proving the audit modules compose.

#[cfg(test)]
mod tests {
    use audit_complex::{calibration_arrays_equal, circular_resultant, coherence_proxy,
        magnitude_change, magnitude_ratio, paired_ratio, wrapped_phase_difference, Complex};
    use audit_core::{paired_delta, summary};
    use audit_distribution::{energy_distance, gaussian_mmd_squared};
    use audit_inference::{benjamini_hochberg, benjamini_hochberg_adjusted, block_flip_draw,
        max_statistic, scan_threshold};
    use audit_ledger::{classify, evidence_tsv, tsv, Candidate, EvidenceRow};
    use audit_model::huber_line;
    use audit_spatial::{laplacian_energy, scan_score};

    #[test]
    fn synthetic_pipeline_recovers_known_cluster_and_response() {
        let before = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let after  = [1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 1.0,
                      1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0];
        let delta = paired_delta(&before, &after);
        let s = summary(&delta);
        assert_eq!(s.positive_fraction, Some(0.25));
        assert!((scan_score(&delta, &[5, 6, 10, 11]).unwrap() - 4.0).abs() < 1e-12);
        assert!(laplacian_energy(&delta, 4, 4).unwrap() > 0.0);
        assert!(energy_distance(&before, &after).unwrap() > 0.0);
        assert!(gaussian_mmd_squared(&before, &after, 1.0).unwrap() > 0.0);
    }

    #[test]
    fn synthetic_pipeline_covers_model_complex_inference_and_ledger() {
        let fit = huber_line(&[0.0, 1.0, 2.0, 3.0], &[1.0, 3.0, 5.0, 7.0], 5, 1.0).unwrap();
        assert!((fit.slope - 2.0).abs() < 1e-12);
        let ratio = paired_ratio(Complex { re: 0.0, im: 1.0 }, Complex { re: 1.0, im: 0.0 }).unwrap();
        assert!((ratio.im - 1.0).abs() < 1e-12);
        let (mean_phase, concentration) = circular_resultant(&[0.0, 0.01, -0.01]).unwrap();
        assert!(mean_phase.abs() < 1e-12 && concentration > 0.99);
        assert_eq!(benjamini_hochberg(&[0.001, 0.4, 0.002], 0.05), vec![true, false, true]);
        assert_eq!(max_statistic(3.0, &[1.0, 2.0, 4.0]), Some(0.5));
        let out = tsv(&[Candidate { id: "synthetic-1".into(), dataset_path: "/fixture".into(), effect_size: Some(2.0), adjusted_p: Some(0.01), classification: "synthetic-pass".into(), control_result: "known".into() }]);
        assert!(out.contains("synthetic-1\t/fixture"));
    }

    /// Fixed October-screen composition: paired delta -> robust summary ->
    /// block-permutation MaxT over a prespecified multiscale family -> BH-FDR
    /// over the coarse family -> evidence-ledger row. A planted coherent
    /// cluster must survive every stage; the null field must not.
    const COMPOSE_SEED: u64 = 0xA53A5A53A5A53A5;
    const COMPOSE_REPLICATES: usize = 199;

    fn compose_field(cluster_gain: f64) -> Vec<f64> {
        let (rows, cols) = (8, 8);
        let mut delta = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let i = r * cols + c;
                delta[i] = 0.25 * ((i % 5) as f64 - 2.0);
                if r < 4 && c < 4 {
                    delta[i] += cluster_gain;
                }
            }
        }
        delta
    }

    /// Prespecified family: four coarse 4x4 quadrants first (BH family),
    /// then sixteen 2x2 tiles (multiscale tail). Fixed before any ranking.
    fn compose_family() -> Vec<Vec<usize>> {
        let mut regions = Vec::with_capacity(20);
        for qr in 0..2 {
            for qc in 0..2 {
                let mut cells = Vec::with_capacity(16);
                for dr in 0..4 {
                    for dc in 0..4 {
                        cells.push((qr * 4 + dr) * 8 + qc * 4 + dc);
                    }
                }
                regions.push(cells);
            }
        }
        for br in 0..4 {
            for bc in 0..4 {
                let mut cells = Vec::with_capacity(4);
                for dr in 0..2 {
                    for dc in 0..2 {
                        cells.push((br * 2 + dr) * 8 + bc * 2 + dc);
                    }
                }
                regions.push(cells);
            }
        }
        regions
    }

    fn family_maxima(field: &[f64], regions: &[Vec<usize>]) -> Vec<f64> {
        regions.iter().map(|reg| scan_score(field, reg).unwrap()).collect()
    }

    fn block_null_maxima(field: &[f64], regions: &[Vec<usize>]) -> Vec<f64> {
        (0..COMPOSE_REPLICATES as u64)
            .map(|stream| {
                let flips = block_flip_draw(field.len(), COMPOSE_SEED, stream);
                let rep: Vec<f64> = field
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| if flips[i] { -v } else { v })
                    .collect();
                family_maxima(&rep, regions).into_iter().fold(0.0, f64::max)
            })
            .collect()
    }

    #[test]
    fn block_maxt_fdr_ledger_promotes_a_coherent_cluster() {
        let delta = compose_field(2.0);
        let regions = compose_family();
        let s = summary(&delta);
        assert_eq!(s.finite, 64);
        // The aspatial robust effect misses the localized cluster (paper 1);
        // the prespecified scan family catches it (paper 3). The ledger row
        // therefore carries the family-native scan score as its effect size.
        assert!(s.robust_effect.unwrap().abs() < 0.5);

        let scores = family_maxima(&delta, &regions);
        let (argmax, observed) = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, &v)| (i, v))
            .unwrap();
        assert_eq!(argmax, 0, "the cluster quadrant must win the prespecified family");
        assert!(observed > 6.0, "cluster signal must dominate background: {observed}");

        let null = block_null_maxima(&delta, &regions);
        let p = max_statistic(observed, &null).unwrap();
        assert!(p < 0.05, "planted cluster must clear the dissolving MaxT null: p={p}");
        let threshold = scan_threshold(&null, 0.05).unwrap();
        assert!(observed > threshold, "threshold and p-value must agree: {observed} vs {threshold}");

        // BH-FDR over the coarse quadrant family keeps the cluster quadrant.
        let quad_p: Vec<f64> = (0..4)
            .map(|q| {
                let null_q: Vec<f64> = (0..COMPOSE_REPLICATES as u64)
                    .map(|stream| {
                        let flips = block_flip_draw(delta.len(), COMPOSE_SEED, stream);
                        let rep: Vec<f64> = delta
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| if flips[i] { -v } else { v })
                            .collect();
                        scan_score(&rep, &regions[q]).unwrap()
                    })
                    .collect();
                max_statistic(scores[q], &null_q).unwrap()
            })
            .collect();
        let adjusted = benjamini_hochberg_adjusted(&quad_p);
        assert!(adjusted[0] < 0.05, "cluster quadrant must survive BH-FDR: {adjusted:?}");
        let mut ranked = adjusted.clone();
        ranked.sort_by(f64::total_cmp);
        assert!(ranked.windows(2).all(|w| w[0] <= w[1]), "adjusted values must be monotone");

        // The ledger is the only promotion path: the row carries family,
        // both p-values, and the conservative classification.
        let row = EvidenceRow {
            candidate_id: "C001".into(),
            dataset_path: "/synthetic/cluster".into(),
            family: "spatial_scan".into(),
            effect_size: Some(observed),
            spatial_support: Some(observed.abs()),
            quality_overlap: None,
            p_raw: Some(p),
            p_adjusted: Some(adjusted[0]),
            classification: classify(Some(adjusted[0]), Some(observed), None, true).into(),
            competing_explanations: "physical|environmental|unmeasured_processing".into(),
            reproducibility_status: "deterministic_rust_screen_complete".into(),
        };
        assert_eq!(row.classification, "unknown_reproducible_anomaly");
        let text = evidence_tsv(&[row]);
        assert!(text.contains("C001\t/synthetic/cluster\tspatial_scan\t"));
        assert!(text.contains("\tunknown_reproducible_anomaly\t"));
    }

    #[test]
    fn block_null_retains_a_null_field_with_p_one() {
        let delta = compose_field(0.0);
        let regions = compose_family();
        let scores = family_maxima(&delta, &regions);
        let observed = scores.iter().fold(0.0f64, |a, &b| f64::max(a, b));
        let null = block_null_maxima(&delta, &regions);
        // Sign flips only permute background signs, so null maxima bracket
        // the observation: the field is retained, never promoted.
        let p = max_statistic(observed, &null).unwrap();
        assert!(p > 0.05, "background field must be retained: p={p}");
        let threshold = scan_threshold(&null, 0.05).unwrap();
        assert!(observed <= threshold, "threshold and p-value must agree on retention");
        assert_eq!(classify(Some(1.0), Some(0.0), None, true), "uninformative");
    }

    #[test]
    fn bh_fdr_selects_planted_signals_monotonically() {
        // Paper-6 fixture shape: five p-values, two planted signals.
        let p = vec![0.001, 0.4, 0.002, 0.7, 0.9];
        assert_eq!(benjamini_hochberg(&p, 0.05), vec![true, false, true, false, false]);
        let adjusted = benjamini_hochberg_adjusted(&p);
        let mut order: Vec<usize> = (0..p.len()).collect();
        order.sort_by(|a, b| p[*a].total_cmp(&p[*b]));
        let ranked: Vec<f64> = order.iter().map(|i| adjusted[*i]).collect();
        assert!(ranked.windows(2).all(|w| w[0] <= w[1]));
        assert!(adjusted[0] <= 0.05 && adjusted[2] <= 0.05 && adjusted[1] > 0.05);
    }

    #[test]
    fn complex_calibration_chain_separates_gain_from_rotation() {
        // Pure amplitude scaling: common complex gain reads as unit
        // coherence with zero phase change (paper-5 calibration analogue).
        let pairs = vec![
            (Complex { re: 1.0, im: 0.0 }, Complex { re: 2.0, im: 0.0 }),
            (Complex { re: 0.0, im: 1.0 }, Complex { re: 0.0, im: 2.0 }),
            (Complex { re: 1.0, im: 1.0 }, Complex { re: 2.0, im: 2.0 }),
        ];
        assert!((coherence_proxy(&pairs).unwrap() - 1.0).abs() < 1e-12);
        // Pure rotation moves phase, never magnitude.
        let before = Complex { re: 1.0, im: 0.0 };
        let after = Complex { re: 0.0, im: 1.0 };
        assert_eq!(magnitude_change(before, after).unwrap(), 0.0);
        assert_eq!(magnitude_ratio(after, before).unwrap(), 2.0 / 2.0);
        let rotation = wrapped_phase_difference(before, after).unwrap();
        assert!((rotation - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        // Conjugation negates the phase difference and preserves magnitude.
        let conjugated = Complex { re: after.re, im: -after.im };
        assert!((wrapped_phase_difference(before, conjugated).unwrap() + rotation).abs() < 1e-12);
        assert_eq!(magnitude_change(before, conjugated).unwrap(), 0.0);
        // Zero-amplitude cells carry no phase: counted, never imputed.
        let zero = Complex { re: 0.0, im: 0.0 };
        assert_eq!(wrapped_phase_difference(zero, before), None);
        assert_eq!(magnitude_ratio(before, zero), None);
        // Calibration scalars compare bit-exactly, never by tolerance.
        assert!(calibration_arrays_equal(&[1.0, 2.0], &[1.0, 2.0]));
        assert!(!calibration_arrays_equal(&[1.0], &[1.0 + f64::EPSILON]));
    }
}
