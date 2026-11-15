//! Complex magnitude/phase primitives without external numeric dependencies.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex { pub re: f64, pub im: f64 }

pub fn magnitude(z: Complex) -> f64 { z.re.hypot(z.im) }
pub fn phase(z: Complex) -> f64 { z.im.atan2(z.re) }
pub fn paired_ratio(after: Complex, before: Complex) -> Option<Complex> {
    let den = before.re*before.re + before.im*before.im;
    (den > 0.0 && den.is_finite()).then_some(Complex { re:(after.re*before.re+after.im*before.im)/den, im:(after.im*before.re-after.re*before.im)/den })
}
pub fn circular_resultant(phases: &[f64]) -> Option<(f64, f64)> {
    if phases.is_empty() { return None; }
    let (s,c) = phases.iter().fold((0.0,0.0), |(s,c),p|(s+p.sin(),c+p.cos()));
    let n=phases.len() as f64; Some((s.atan2(c), (s*s+c*c).sqrt()/n))
}

/// Wrap a phase difference to `(-pi, pi]`. `None` for non-finite input.
pub fn wrap_phase(d: f64) -> Option<f64> {
    if !d.is_finite() {
        return None;
    }
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut w = d.rem_euclid(two_pi);
    if w > std::f64::consts::PI {
        w -= two_pi;
    }
    // rem_euclid maps -pi to pi; keep the branch cut at (-pi, pi].
    if w <= -std::f64::consts::PI {
        w += two_pi;
    }
    Some(w)
}

fn magnitude_finite(z: Complex) -> Option<f64> {
    let m = magnitude(z);
    m.is_finite().then_some(m)
}

/// Change in magnitude `|after| - |before|`. `None` when either side is
/// non-finite. A pure phase rotation reads exactly zero here by construction.
pub fn magnitude_change(before: Complex, after: Complex) -> Option<f64> {
    Some(magnitude_finite(after)? - magnitude_finite(before)?)
}

/// Amplitude gain `|after| / |before|`. `None` for a zero or non-finite
/// denominator — a gain through zero is uninformative, not infinite.
pub fn magnitude_ratio(after: Complex, before: Complex) -> Option<f64> {
    let den = magnitude_finite(before)?;
    if !(den > 0.0) {
        return None;
    }
    let num = magnitude_finite(after)?;
    Some(num / den)
}

/// Wrapped paired phase change `arg(after) - arg(before)` in `(-pi, pi]`.
/// `None` when either phasor has zero magnitude (phase undefined) or a
/// non-finite component.
pub fn wrapped_phase_difference(before: Complex, after: Complex) -> Option<f64> {
    let mb = magnitude_finite(before)?;
    let ma = magnitude_finite(after)?;
    if !(mb > 0.0) || !(ma > 0.0) {
        return None;
    }
    wrap_phase(phase(after) - phase(before))
}

/// Circular mean over finite phases only. Returns `(mean_direction,
/// resultant_length)`. `None` when no phase is finite. Non-finite entries
/// are skipped, never imputed.
pub fn circular_mean_finite(phases: &[f64]) -> Option<(f64, f64)> {
    let finite: Vec<f64> = phases.iter().copied().filter(|p| p.is_finite()).collect();
    if finite.is_empty() {
        return None;
    }
    circular_resultant(&finite)
}

/// Coherence proxy over paired phasors:
///
/// ```text
/// gamma = |sum conj(before) * after| / sqrt(sum|before|^2 * sum|after|^2)
/// ```
///
/// Unity for a common complex gain (pure calibration-like scaling),
/// near zero for decorrelated pairs. Non-finite pairs are skipped;
/// `None` when fewer than one finite pair survives or either energy is zero.
pub fn coherence_proxy(pairs: &[(Complex, Complex)]) -> Option<f64> {
    let mut re = 0.0;
    let mut im = 0.0;
    let mut eb = 0.0;
    let mut ea = 0.0;
    let mut n = 0usize;
    for (before, after) in pairs {
        if ![before.re, before.im, after.re, after.im].iter().all(|v| v.is_finite()) {
            continue;
        }
        // conj(before) * after
        re += before.re * after.re + before.im * after.im;
        im += before.re * after.im - before.im * after.re;
        eb += before.re * before.re + before.im * before.im;
        ea += after.re * after.re + after.im * after.im;
        n += 1;
    }
    if n == 0 || !(eb > 0.0) || !(ea > 0.0) {
        return None;
    }
    Some((re * re + im * im).sqrt() / (eb * ea).sqrt())
}

/// Spatial phase roughness: sum of squared wrapped differences over rook
/// neighbour pairs that are jointly finite. A smooth calibration-like
/// ramp has low energy; speckle-like phase has high energy at the same
/// circular mean. `None` on shape mismatch or empty grid.
pub fn wrapped_neighbor_energy(phases: &[f64], rows: usize, cols: usize) -> Option<f64> {
    if rows.checked_mul(cols)? != phases.len() || rows == 0 || cols == 0 {
        return None;
    }
    let mut total = 0.0;
    for r in 0..rows {
        for c in 0..cols {
            let i = r * cols + c;
            if !phases[i].is_finite() {
                continue;
            }
            if c + 1 < cols && phases[i + 1].is_finite() {
                if let Some(w) = wrap_phase(phases[i + 1] - phases[i]) {
                    total += w * w;
                }
            }
            if r + 1 < rows && phases[i + cols].is_finite() {
                if let Some(w) = wrap_phase(phases[i + cols] - phases[i]) {
                    total += w * w;
                }
            }
        }
    }
    Some(total)
}

/// Exact equality of two calibration arrays (byte-identical scalars read
/// as `f64`, NaN payloads compared by bits). Calibration consistency is
/// an equality check, not a tolerance fit: any difference is reported.
pub fn calibration_arrays_equal(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[cfg(test)]
mod tests { use super::*;
    #[test] fn phase_and_magnitude_are_consistent() { let z=Complex{re:3.0,im:4.0}; assert_eq!(magnitude(z),5.0); assert!((phase(z)-4.0_f64.atan2(3.0)).abs()<1e-12); }
    #[test] fn ratio_recovers_amplitude() { let r=paired_ratio(Complex{re:2.0,im:0.0},Complex{re:1.0,im:0.0}).unwrap(); assert_eq!(r,Complex{re:2.0,im:0.0}); }
    #[test] fn pure_rotation_moves_phase_not_magnitude() {
        let before = Complex { re: 1.0, im: 0.0 };
        let after = Complex { re: 0.0, im: 1.0 }; // +90 degrees
        assert_eq!(magnitude_change(before, after).unwrap(), 0.0);
        let d = wrapped_phase_difference(before, after).unwrap();
        assert!((d - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert_eq!(wrap_phase(3.0 * std::f64::consts::PI).unwrap(), std::f64::consts::PI);
    }
    #[test] fn degenerate_phasors_are_uninformative_not_zero() {
        let zero = Complex { re: 0.0, im: 0.0 };
        let unit = Complex { re: 1.0, im: 0.0 };
        assert_eq!(wrapped_phase_difference(zero, unit), None);
        assert_eq!(magnitude_ratio(unit, zero), None);
        assert_eq!(circular_mean_finite(&[]), None);
        assert_eq!(circular_mean_finite(&[f64::NAN]), None);
    }
    #[test] fn common_gain_gives_unit_coherence() {
        // after = 2 * before for every pair: pure amplitude scaling.
        let pairs = vec![
            (Complex { re: 1.0, im: 0.0 }, Complex { re: 2.0, im: 0.0 }),
            (Complex { re: 0.0, im: 1.0 }, Complex { re: 0.0, im: 2.0 }),
            (Complex { re: 1.0, im: 1.0 }, Complex { re: 2.0, im: 2.0 }),
        ];
        assert!((coherence_proxy(&pairs).unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(coherence_proxy(&[]), None);
    }
    #[test] fn phase_energy_separates_ramp_from_speckle() {
        let ramp = vec![0.0, 0.1, 0.2, 0.3]; // smooth 2x2
        let checker = vec![0.0, std::f64::consts::PI, std::f64::consts::PI, 0.0];
        let e_ramp = wrapped_neighbor_energy(&ramp, 2, 2).unwrap();
        let e_check = wrapped_neighbor_energy(&checker, 2, 2).unwrap();
        assert!(e_check > e_ramp);
        assert_eq!(wrapped_neighbor_energy(&[0.0; 3], 2, 2), None);
    }
    #[test] fn calibration_equality_is_bit_exact() {
        assert!(calibration_arrays_equal(&[1.0, 2.0], &[1.0, 2.0]));
        assert!(!calibration_arrays_equal(&[1.0], &[1.0, 2.0]));
        assert!(!calibration_arrays_equal(&[1.0], &[1.0 + f64::EPSILON]));
    }
}
