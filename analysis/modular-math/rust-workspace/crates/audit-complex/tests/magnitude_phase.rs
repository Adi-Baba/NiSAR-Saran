//! Magnitude/phase separation fixtures: a pure rotation moves phase but
//! not magnitude, a common gain is fully coherent, and degenerate
//! phasors are uninformative rather than zero.

use audit_complex::*;
use std::f64::consts::{FRAC_PI_2, PI};

fn phasor(mag: f64, ph: f64) -> Complex {
    Complex { re: mag * ph.cos(), im: mag * ph.sin() }
}

#[test]
fn rotation_and_gain_separate_cleanly() {
    let before = phasor(2.0, 0.3);
    // Pure +90-degree rotation at fixed magnitude.
    let rotated = phasor(2.0, 0.3 + FRAC_PI_2);
    assert!(magnitude_change(before, rotated).unwrap().abs() < 1e-12);
    let d = wrapped_phase_difference(before, rotated).unwrap();
    assert!((d - FRAC_PI_2).abs() < 1e-12);
    // Pure x3 gain at fixed phase.
    let gained = phasor(6.0, 0.3);
    assert!((magnitude_change(before, gained).unwrap() - 4.0).abs() < 1e-12);
    assert!((magnitude_ratio(gained, before).unwrap() - 3.0).abs() < 1e-12);
    assert!(wrapped_phase_difference(before, gained).unwrap().abs() < 1e-12);
}

#[test]
fn circular_mean_concentrates_aligned_phases() {
    let aligned = vec![0.1, 0.0, -0.1, 0.05];
    let (mean, conc) = circular_mean_finite(&aligned).unwrap();
    assert!(mean.abs() < 0.1);
    assert!(conc > 0.99, "aligned phases must concentrate: {conc}");
    let spread: Vec<f64> = (0..8).map(|k| k as f64 * PI / 4.0).collect();
    let (_, conc_spread) = circular_mean_finite(&spread).unwrap();
    assert!(conc_spread < conc);
    // Non-finite entries are skipped, not imputed.
    let (m2, _) = circular_mean_finite(&[0.0, f64::NAN, 0.0]).unwrap();
    assert!(m2.abs() < 1e-12);
}

#[test]
fn coherence_marks_scaling_vs_decorrelation() {
    let pairs: Vec<(Complex, Complex)> = (0..6)
        .map(|k| {
            let b = phasor(1.0 + k as f64 * 0.1, 0.2 * k as f64);
            (b, phasor(2.0 * (1.0 + k as f64 * 0.1), 0.2 * k as f64))
        })
        .collect();
    assert!((coherence_proxy(&pairs).unwrap() - 1.0).abs() < 1e-12);
    // Opposite phasors cancel: decorrelated pairs score near zero.
    let anti = vec![
        (phasor(1.0, 0.0), phasor(1.0, 0.0)),
        (phasor(1.0, 0.0), phasor(1.0, PI)),
    ];
    assert!(coherence_proxy(&anti).unwrap() < 0.5);
    assert_eq!(coherence_proxy(&[]), None);
}

#[test]
fn calibration_screen_uses_equality_not_tolerance() {
    assert!(calibration_arrays_equal(&[1.0, 2.5, 3.0], &[1.0, 2.5, 3.0]));
    assert!(!calibration_arrays_equal(&[1.0, 2.5], &[1.0, 2.5 + 1e-15]));
    // Wrapped branch cut: +pi and -pi are the same direction.
    assert!(wrap_phase(PI).unwrap().abs() - PI < 1e-12);
    assert!((wrap_phase(-PI).unwrap() - PI).abs() < 1e-12);
    assert_eq!(wrap_phase(f64::INFINITY), None);
}
