//! Stable evidence-ledger rows and TSV serialization.
//!
//! The ledger is the only promotion path: a candidate enters the final
//! report only as an [`EvidenceRow`], and every screened path — including
//! null, unsupported, and failed rows — stays in the table with `NA`
//! statistics instead of being dropped. Column order is frozen by
//! [`EVIDENCE_COLUMNS`] and pinned by the integration tests, so the
//! manuscript schema (paper 6) and the runner output cannot drift apart.

#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub id: String,
    pub dataset_path: String,
    pub effect_size: Option<f64>,
    pub adjusted_p: Option<f64>,
    pub classification: String,
    pub control_result: String,
}

impl Candidate {
    pub fn tsv_header() -> &'static str { "candidate_id\tdataset_path\teffect_size\tadjusted_p\tclassification\tcontrol_result" }
    pub fn to_tsv(&self) -> String { format!("{}\t{}\t{}\t{}\t{}\t{}", self.id, self.dataset_path, fmt(self.effect_size), fmt(self.adjusted_p), self.classification, self.control_result) }
}
fn fmt(v: Option<f64>) -> String { v.map(|x| format!("{x:.12e}")).unwrap_or_else(|| "NA".into()) }

pub fn tsv(rows: &[Candidate]) -> String {
    let mut out = String::from(Candidate::tsv_header()); out.push('\n');
    for row in rows { out.push_str(&row.to_tsv()); out.push('\n'); }
    out
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceRow {
    pub candidate_id: String,
    pub dataset_path: String,
    pub family: String,
    pub effect_size: Option<f64>,
    pub spatial_support: Option<f64>,
    pub quality_overlap: Option<f64>,
    pub p_raw: Option<f64>,
    pub p_adjusted: Option<f64>,
    pub classification: String,
    pub competing_explanations: String,
    pub reproducibility_status: String,
}

/// Frozen evidence-ledger column order. `header()` must equal these joined
/// with tabs; the integration tests pin that invariant.
pub const EVIDENCE_COLUMNS: [&str; 11] = [
    "candidate_id", "dataset_path", "family", "effect_size", "spatial_support",
    "quality_overlap", "p_raw", "p_adjusted", "classification",
    "competing_explanations", "reproducibility_status",
];

impl EvidenceRow {
    pub fn header() -> &'static str {
        "candidate_id\tdataset_path\tfamily\teffect_size\tspatial_support\tquality_overlap\tp_raw\tp_adjusted\tclassification\tcompeting_explanations\treproducibility_status"
    }
    pub fn to_tsv(&self) -> String {
        format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            clean(&self.candidate_id), clean(&self.dataset_path), clean(&self.family),
            fmt(self.effect_size), fmt(self.spatial_support), fmt(self.quality_overlap),
            fmt(self.p_raw), fmt(self.p_adjusted), clean(&self.classification),
            clean(&self.competing_explanations), clean(&self.reproducibility_status))
    }
}

fn clean(value: &str) -> String { value.replace(['\t', '\n', '\r'], " ") }

pub fn evidence_tsv(rows: &[EvidenceRow]) -> String {
    let mut out = String::from(EvidenceRow::header()); out.push('\n');
    for row in rows { out.push_str(&row.to_tsv()); out.push('\n'); }
    out
}

/// Conservative classification used by the first complete audit pass.
pub fn classify(adjusted_p: Option<f64>, robust_effect: Option<f64>, quality_overlap: Option<f64>, supported: bool) -> &'static str {
    if !supported { return "uninformative_or_unsupported"; }
    let significant = adjusted_p.map(|p| p <= 0.05).unwrap_or(false);
    let material = robust_effect.map(|e| e.abs() >= 0.5).unwrap_or(false);
    if !(significant && material) { return "uninformative"; }
    if quality_overlap.map(|x| x >= 0.5).unwrap_or(false) { return "quality_geometry_consistent"; }
    "unknown_reproducible_anomaly"
}

#[cfg(test)]
mod tests { use super::*;
    #[test] fn ledger_is_stable_tsv() { let s=tsv(&[Candidate{id:"x".into(),dataset_path:"/a".into(),effect_size:Some(1.0),adjusted_p:None,classification:"unknown".into(),control_result:"pending".into()}]); assert!(s.contains("x\t/a\t1.000000000000e0\tNA")); }
    #[test] fn classification_is_conservative() {
        assert_eq!(classify(Some(0.01), Some(1.0), Some(0.8), true), "quality_geometry_consistent");
        assert_eq!(classify(Some(0.01), Some(1.0), Some(0.1), true), "unknown_reproducible_anomaly");
        assert_eq!(classify(None, None, None, false), "uninformative_or_unsupported");
    }
}
