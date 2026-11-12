//! The physical model of the survey galaxies: cosmology plus k-correction, e-correction
//! (luminosity evolution) and P(z) (number density evolution), following Cole (2011) §3.
//!
//! Apparent and absolute magnitudes are related by (Cole 2011 eq. 32)
//!
//! `m = M + DM(z) + k(z) + e(z) - [k(z_ref) + e(z_ref)]`
//!
//! and Phi*(z) = P(z) Phi*(z_ref).

use serde::Deserialize;

use crate::cosmology::Cosmology;

/// k-correction applied to every galaxy (an average for the population).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum KCorrection {
    /// k(z) = 0.
    #[default]
    None,
    /// k(z) = sum_i coeffs[i] * (z - z0)^i.
    ///
    /// GAMA r-band (Cole Fortran `user_routines.f90`):
    /// coeffs = [0.20848, 1.0226, 0.52366, 3.5902, 2.3843], z0 = 0.2.
    Polynomial { coeffs: Vec<f64>, z0: f64 },
}

impl KCorrection {
    pub fn k(&self, z: f64) -> f64 {
        match self {
            KCorrection::None => 0.,
            KCorrection::Polynomial { coeffs, z0 } => {
                let x = z - z0;
                coeffs.iter().rev().fold(0., |acc, &c| acc * x + c)
            }
        }
    }
}

/// Luminosity evolution e(z) = -q z (galaxies were brighter in the past for q > 0).
///
/// GAMA (Cole Fortran `ecorr`, u = 0): q = 1.75.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ECorrection {
    #[serde(default)]
    pub q: f64,
}

impl ECorrection {
    pub fn e(&self, z: f64) -> f64 {
        -self.q * z
    }
}

/// Number density evolution P(z), with Phi*(z) = P(z) Phi*(z_ref).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum DensityEvolution {
    /// P(z) = 1.
    #[default]
    None,
    /// P(z) = exp(a z). Cole Fortran GAMA default is a = 0.1.
    Exp { a: f64 },
}

impl DensityEvolution {
    pub fn p(&self, z: f64) -> f64 {
        match self {
            DensityEvolution::None => 1.,
            DensityEvolution::Exp { a } => (a * z).exp(),
        }
    }
}

/// Everything needed to convert between apparent magnitude, absolute magnitude, redshift and
/// (P-weighted) volume.
#[derive(Debug, Clone)]
pub struct Model {
    pub cosmology: Cosmology,
    pub kcorr: KCorrection,
    pub ecorr: ECorrection,
    pub density: DensityEvolution,
    /// Reference redshift at which k + e = 0 and P = 1 by definition.
    pub zref: f64,
}

impl Model {
    /// A model with no k-correction, no evolution and z_ref = 0.
    pub fn new(cosmology: Cosmology) -> Self {
        Model {
            cosmology,
            kcorr: KCorrection::None,
            ecorr: ECorrection::default(),
            density: DensityEvolution::None,
            zref: 0.,
        }
    }

    /// Combined k+e correction relative to z_ref (Fortran `kpluse`).
    pub fn kpluse(&self, z: f64) -> f64 {
        self.kcorr.k(z) + self.ecorr.e(z) - (self.kcorr.k(self.zref) + self.ecorr.e(self.zref))
    }

    /// P(z)/P(z_ref).
    pub fn p_ratio(&self, z: f64) -> f64 {
        self.density.p(z) / self.density.p(self.zref)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gama_polynomial_matches_fortran() {
        let k = KCorrection::Polynomial {
            coeffs: vec![0.20848, 1.0226, 0.52366, 3.5902, 2.3843],
            z0: 0.2,
        };
        // Direct evaluation of the Fortran loop at z = 0.35
        let x: f64 = 0.15;
        let expected = 0.20848 + 1.0226 * x + 0.52366 * x.powi(2) + 3.5902 * x.powi(3)
            + 2.3843 * x.powi(4);
        assert!((k.k(0.35) - expected).abs() < 1e-12);
    }

    #[test]
    fn kpluse_zero_at_zref() {
        let mut m = Model::new(Cosmology {
            omega_m: 0.3,
            omega_k: 0.,
            omega_l: 0.7,
            h0: 100.,
        });
        m.kcorr = KCorrection::Polynomial {
            coeffs: vec![0.1, 1.0],
            z0: 0.,
        };
        m.ecorr = ECorrection { q: 1.75 };
        m.zref = 0.1;
        assert!(m.kpluse(0.1).abs() < 1e-14);
        assert!((m.kpluse(0.3) - (0.2 - 1.75 * 0.2)).abs() < 1e-12);
    }
}
