//! Ledger schema and classification-boundary integration tests.
//!
//! These pin the paper-6 contract from outside the crate: frozen column
//! order, byte-stable TSV, tab/newline sanitization, and the exact
//! promotion boundaries used by the October run.

use audit_ledger::{classify, evidence_tsv, EvidenceRow, EVIDENCE_COLUMNS};

fn row(id: &str) -> EvidenceRow {
    EvidenceRow {
        candidate_id: id.into(),
        dataset_path: "/science/LSAR/GCOV/grids/frequencyB/HHHH".into(),
        family: "paired_dataset".into(),
        effect_size: Some(1.45141969955),
        spatial_support: Some(33.34056472445),
        quality_overlap: None,
        p_raw: Some(0.001),
        p_adjusted: Some(0.00688),
        classification: "unknown_reproducible_anomaly".into(),
        competing_explanations: "physical|environmental|unmeasured_processing".into(),
        reproducibility_status: "deterministic_rust_screen_complete".into(),
    }
}

#[test]
fn header_matches_the_frozen_column_list() {
    assert_eq!(EvidenceRow::header(), EVIDENCE_COLUMNS.join("\t"));
    assert_eq!(EVIDENCE_COLUMNS.len(), 11);
}

#[test]
fn ledger_tsv_is_byte_stable_and_complete() {
    let rows = vec![row("C001"), row("C002")];
    let first = evidence_tsv(&rows);
    assert_eq!(first, evidence_tsv(&rows));
    let lines: Vec<&str> = first.lines().collect();
    assert_eq!(lines.len(), 3);
    assert!(lines[1].starts_with("C001\t/science/LSAR/GCOV/grids/frequencyB/HHHH\tpaired_dataset\t"));
    assert!(lines[1].contains("\tunknown_reproducible_anomaly\tphysical|environmental|unmeasured_processing\tdeterministic_rust_screen_complete"));
    // No row is ever dropped: null and unsupported rows stay with NA fields.
    let mut null = row("C099");
    null.effect_size = None;
    null.p_raw = None;
    null.p_adjusted = None;
    null.classification = "uninformative_or_unsupported".into();
    let text = evidence_tsv(&[null]);
    assert!(text.contains("C099\t/science/LSAR/GCOV/grids/frequencyB/HHHH\tpaired_dataset\tNA\t"));
}

#[test]
fn control_characters_cannot_break_columns() {
    let mut dirty = row("C003");
    dirty.dataset_path = "/a\tb\nc\rd".into();
    let text = evidence_tsv(&[dirty]);
    assert_eq!(text.lines().count(), 2);
    assert!(text.contains("/a b c d"));
}

#[test]
fn promotion_boundaries_match_the_october_screen() {
    // Significant and material with no quality overlap: anomaly candidate.
    assert_eq!(classify(Some(0.01), Some(1.0), Some(0.1), true), "unknown_reproducible_anomaly");
    // Exact thresholds are inclusive: p <= 0.05, |effect| >= 0.5, overlap >= 0.5.
    assert_eq!(classify(Some(0.05), Some(0.5), None, true), "unknown_reproducible_anomaly");
    assert_eq!(classify(Some(0.01), Some(1.0), Some(0.5), true), "quality_geometry_consistent");
    // Just outside any single gate demotes to uninformative.
    assert_eq!(classify(Some(0.0500001), Some(1.0), None, true), "uninformative");
    assert_eq!(classify(Some(0.01), Some(0.4999), None, true), "uninformative");
    // NaN statistics never promote and never panic.
    assert_eq!(classify(Some(f64::NAN), Some(1.0), None, true), "uninformative");
    assert_eq!(classify(Some(0.01), Some(f64::NAN), None, true), "uninformative");
    assert_eq!(classify(None, None, None, true), "uninformative");
    // Unsupported representations are never promoted, however small the p-value.
    assert_eq!(classify(Some(0.0001), Some(9.9), Some(0.0), false), "uninformative_or_unsupported");
}
