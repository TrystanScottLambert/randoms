//! How clones are distributed within each galaxy's accessible (P-weighted) volume.
//!
//! * [`Window::Uniform`]: Cole (2011) §4. Uniform in V between V(z_min,i) and V(z_max,i).
//! * [`Window::Gaussian`]: Farrow et al. (2015) §3.1.2. A Gaussian in volume centred on the
//!   parent galaxy's own volume, truncated at `truncate` sigma, and **reflected** (not
//!   truncated) at the edges of the galaxy's accessible volume.

use rand::Rng;
use rand_distr::StandardNormal;

/// Width of the Gaussian window.
#[derive(Debug, Clone, Copy)]
pub enum Sigma {
    /// Absolute width in the same units as the P-weighted survey volume ((Mpc/h)^3 for
    /// h0 = 100) *within the survey solid angle*. Farrow et al. (2015) used 3.5e6 (Mpc/h)^3 for
    /// GAMA II.
    Volume(f64),
    /// Width as a fraction of the total P-weighted survey volume between zmin and zmax.
    FractionOfSurvey(f64),
}

#[derive(Debug, Clone, Copy)]
pub enum Window {
    Uniform,
    Gaussian { sigma: Sigma, truncate: f64 },
}

/// Reflect `x` back into `[lo, hi]` at both boundaries, as many times as needed. This is
/// equivalent to folding a periodic extension with period 2(hi - lo).
pub fn fold(x: f64, lo: f64, hi: f64) -> f64 {
    let l = hi - lo;
    if l <= 0. {
        return lo;
    }
    let y = (x - lo).rem_euclid(2. * l);
    lo + if y > l { 2. * l - y } else { y }
}

/// Draw a standard normal deviate truncated to |g| <= truncate.
pub(crate) fn truncated_standard_normal<R: Rng>(truncate: f64, rng: &mut R) -> f64 {
    loop {
        let g: f64 = rng.sample(StandardNormal);
        if g.abs() <= truncate {
            return g;
        }
    }
}

/// Midpoint quadrature of a standard normal truncated at +-truncate, used to compute the
/// fraction of a galaxy's window that falls in each redshift bin.
pub(crate) struct NormalQuadrature {
    pub nodes: Vec<f64>,
    pub weights: Vec<f64>,
}

impl NormalQuadrature {
    pub fn new(truncate: f64, n: usize) -> Self {
        let h = 2. * truncate / n as f64;
        let nodes: Vec<f64> = (0..n).map(|j| -truncate + (j as f64 + 0.5) * h).collect();
        let raw: Vec<f64> = nodes.iter().map(|&t| (-0.5 * t * t).exp()).collect();
        let total: f64 = raw.iter().sum();
        let weights = raw.iter().map(|w| w / total).collect();
        NormalQuadrature { nodes, weights }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn fold_reflects() {
        assert_eq!(fold(0.5, 0., 1.), 0.5);
        assert!((fold(1.2, 0., 1.) - 0.8).abs() < 1e-12);
        assert!((fold(-0.3, 0., 1.) - 0.3).abs() < 1e-12);
        assert!((fold(2.3, 0., 1.) - 0.3).abs() < 1e-12); // two reflections
        assert!((fold(-1.7, 0., 1.) - 0.3).abs() < 1e-12);
        assert_eq!(fold(5., 2., 2.), 2.);
    }

    #[test]
    fn folding_a_wide_window_tends_to_uniform() {
        // A window much wider than the interval should fill it uniformly: no pile-up or
        // deficit at the edges (the failure mode of truncation that Farrow warns about).
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let n = 400_000;
        let mut hist = [0usize; 10];
        for _ in 0..n {
            let x = fold(0.1 + 50. * truncated_standard_normal(2., &mut rng), 0., 1.);
            hist[((x * 10.) as usize).min(9)] += 1;
        }
        let expected = n as f64 / 10.;
        for h in hist {
            assert!((h as f64 / expected - 1.).abs() < 0.02, "{hist:?}");
        }
    }

    #[test]
    fn quadrature_normalised_and_symmetric() {
        let q = NormalQuadrature::new(2., 400);
        assert!((q.weights.iter().sum::<f64>() - 1.).abs() < 1e-12);
        let mean: f64 = q.nodes.iter().zip(&q.weights).map(|(t, w)| t * w).sum();
        assert!(mean.abs() < 1e-12);
    }
}
