//! Tabulated redshift relations on a fine uniform redshift grid (the Rust equivalent of the
//! Fortran `tabulate_distance` / `search_distancemag`).
//!
//! Everything is built from one cumulative trapezoid integration of 1/E(z), so the volumes and
//! distance moduli are mutually consistent.

use crate::model::Model;

pub struct DistanceTable {
    dz: f64,
    /// Redshift grid, uniform from 0 to `zmax`.
    pub z: Vec<f64>,
    /// m - M = DM(z) + kpluse(z). `-inf` at z = 0.
    pub mag_offset: Vec<f64>,
    /// Running maximum of `mag_offset`, used to find the *first* redshift at which a galaxy
    /// crosses a magnitude limit even if k+e makes m(z) non-monotonic.
    offset_cummax: Vec<f64>,
    /// Comoving volume out to z within the survey solid angle.
    pub volume: Vec<f64>,
    /// P-weighted comoving volume, int_0^z P(z')/P(z_ref) dV/dz' dz', within the survey solid
    /// angle. This is the coordinate in which clones are distributed.
    pub pvolume: Vec<f64>,
}

impl DistanceTable {
    /// Tabulate on `n` points between z = 0 and `zmax`, for a survey of `solid_angle_sr`
    /// steradians.
    pub fn new(model: &Model, zmax: f64, solid_angle_sr: f64, n: usize) -> Self {
        assert!(n >= 2, "table needs at least 2 points");
        assert!(zmax > 0., "zmax must be positive");
        let cosmo = &model.cosmology;
        let dh = cosmo.hubble_distance();
        let ok = cosmo.omega_k;
        let dz = zmax / (n - 1) as f64;

        let z: Vec<f64> = (0..n).map(|i| i as f64 * dz).collect();
        let inv_e: Vec<f64> = z.iter().map(|&zi| 1. / cosmo.e_func(zi)).collect();

        // Line-of-sight comoving distance by cumulative trapezoid.
        let mut dc = vec![0.; n];
        for i in 1..n {
            dc[i] = dc[i - 1] + 0.5 * (inv_e[i - 1] + inv_e[i]) * dz * dh;
        }
        // Transverse comoving distance (Hogg 1999 eq. 16).
        let dm: Vec<f64> = dc
            .iter()
            .map(|&d| {
                if ok > 0. {
                    dh / ok.sqrt() * (ok.sqrt() * d / dh).sinh()
                } else if ok < 0. {
                    dh / ok.abs().sqrt() * (ok.abs().sqrt() * d / dh).sin()
                } else {
                    d
                }
            })
            .collect();

        let dvdz: Vec<f64> = (0..n)
            .map(|i| solid_angle_sr * dh * dm[i] * dm[i] * inv_e[i])
            .collect();
        let pdvdz: Vec<f64> = (0..n).map(|i| model.p_ratio(z[i]) * dvdz[i]).collect();

        let mut volume = vec![0.; n];
        let mut pvolume = vec![0.; n];
        for i in 1..n {
            volume[i] = volume[i - 1] + 0.5 * (dvdz[i - 1] + dvdz[i]) * dz;
            pvolume[i] = pvolume[i - 1] + 0.5 * (pdvdz[i - 1] + pdvdz[i]) * dz;
        }

        let mag_offset: Vec<f64> = (0..n)
            .map(|i| {
                let dl = (1. + z[i]) * dm[i];
                5. * dl.log10() + 25. + model.kpluse(z[i])
            })
            .collect();
        let mut offset_cummax = mag_offset.clone();
        for i in 1..n {
            offset_cummax[i] = offset_cummax[i].max(offset_cummax[i - 1]);
        }

        DistanceTable {
            dz,
            z,
            mag_offset,
            offset_cummax,
            volume,
            pvolume,
        }
    }

    pub fn zmax(&self) -> f64 {
        *self.z.last().unwrap()
    }

    fn interp(&self, arr: &[f64], z: f64) -> f64 {
        let n = arr.len();
        let t = (z / self.dz).clamp(0., (n - 1) as f64);
        let i = (t.floor() as usize).min(n - 2);
        let frac = t - i as f64;
        if !arr[i].is_finite() {
            // only the z = 0 distance modulus
            return if frac == 0. { arr[i] } else { arr[i + 1] };
        }
        arr[i] + (arr[i + 1] - arr[i]) * frac
    }

    /// m - M at redshift z.
    pub fn mag_offset_at(&self, z: f64) -> f64 {
        self.interp(&self.mag_offset, z)
    }

    pub fn volume_at(&self, z: f64) -> f64 {
        self.interp(&self.volume, z)
    }

    pub fn pvolume_at(&self, z: f64) -> f64 {
        self.interp(&self.pvolume, z)
    }

    /// Inverse of `pvolume_at`.
    pub fn z_at_pvolume(&self, u: f64) -> f64 {
        let n = self.pvolume.len();
        let k = self.pvolume.partition_point(|&v| v < u);
        if k == 0 {
            return 0.;
        }
        if k >= n {
            return self.zmax();
        }
        let (v0, v1) = (self.pvolume[k - 1], self.pvolume[k]);
        let frac = if v1 > v0 { (u - v0) / (v1 - v0) } else { 0. };
        self.z[k - 1] + frac * self.dz
    }

    /// The first redshift at which a galaxy of absolute magnitude `absmag` becomes fainter than
    /// `mag_limit`. Returns the table's zmax if it never does.
    pub fn limit_z(&self, absmag: f64, mag_limit: f64) -> f64 {
        let target = mag_limit - absmag;
        let n = self.offset_cummax.len();
        let k = self.offset_cummax.partition_point(|&v| v <= target);
        if k >= n {
            return self.zmax();
        }
        if k == 0 {
            return 0.;
        }
        let (lo, hi) = (self.offset_cummax[k - 1], self.offset_cummax[k]);
        if !lo.is_finite() || hi <= lo {
            return self.z[k - 1];
        }
        self.z[k - 1] + self.dz * (target - lo) / (hi - lo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosmology::Cosmology;
    use std::f64::consts::PI;

    fn flat() -> Model {
        Model::new(Cosmology {
            omega_m: 0.3,
            omega_k: 0.,
            omega_l: 0.7,
            h0: 100.,
        })
    }

    #[test]
    fn volumes_match_cosmology() {
        let model = flat();
        let t = DistanceTable::new(&model, 2.0, 4. * PI, 20001);
        for z in [0.05, 0.2, 1.0, 2.0] {
            let v = model.cosmology.comoving_volume(z);
            assert!((t.volume_at(z) / v - 1.).abs() < 1e-5, "z={z}");
            // no density evolution -> pvolume == volume
            assert!((t.pvolume_at(z) / v - 1.).abs() < 1e-5);
            assert!((t.z_at_pvolume(t.pvolume_at(z)) - z).abs() < 1e-6);
        }
    }

    #[test]
    fn distance_modulus_matches_cosmology() {
        let model = flat();
        let t = DistanceTable::new(&model, 4.0, 1., 40001);
        for z in [0.1, 0.3, 1.0, 4.0] {
            assert!((t.mag_offset_at(z) - model.cosmology.distance_modulus(z)).abs() < 1e-5);
        }
    }

    #[test]
    fn limit_z_without_corrections_matches_old_test() {
        // Same case as the old `test_max_z`: m=19.6 at z=0.2 with limit 19.8 -> zmax 0.21713688
        let model = flat();
        let t = DistanceTable::new(&model, 1.0, 1., 20001);
        let absmag = 19.6 - t.mag_offset_at(0.2);
        assert!((t.limit_z(absmag, 19.8) - 0.21713688).abs() < 1e-5);
        let absmag = 19.8 - t.mag_offset_at(0.2);
        assert!((t.limit_z(absmag, 19.8) - 0.2).abs() < 1e-6);
        // Very bright galaxy is visible to the edge of the table
        assert_eq!(t.limit_z(-30., 19.8), 1.0);
    }

    #[test]
    fn limit_z_with_kcorrection_round_trips() {
        let mut model = flat();
        model.kcorr = crate::model::KCorrection::Polynomial {
            coeffs: vec![0.20848, 1.0226, 0.52366, 3.5902, 2.3843],
            z0: 0.2,
        };
        model.ecorr = crate::model::ECorrection { q: 1.75 };
        let t = DistanceTable::new(&model, 0.6, 1., 20001);
        // a galaxy with M chosen so it hits m = 19.65 at z = 0.3
        let absmag = 19.65 - t.mag_offset_at(0.3);
        assert!((t.limit_z(absmag, 19.65) - 0.3).abs() < 1e-5);
    }
}
