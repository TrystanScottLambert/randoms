//! # randoms
//!
//! randoms is a crate to generate random catalogues of galaxies from redshift surveys in order
//! to better model the selction function of these surveys and remove the underlying large-scale
//! structure. This is essential for things like studying clustering and modelling the n(z) for
//! group finding.

pub mod constants;
pub mod cosmology;
pub mod histogram;

use interp::{InterpMode, interp};
use libm::log10;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal as StatrsNormal};
use std::f64::consts::PI;

use crate::cosmology::Cosmology;
use crate::histogram::{arange, calculate_fd, histogram, linspace};

/// Calcuates the redshift at which the current galaxy with magntidue mag and redshift z would be visible.
fn calculate_max_z(mag: f64, z: f64, mag_lim: f64, cosmo: &Cosmology) -> f64 {
    let distance_measured = cosmo.luminosity_distance(z) * 1e6;
    let max_distance = 10_f64.powf((mag_lim - mag + 5. * log10(distance_measured)) / 5.);
    cosmo.inverse_lumdist(max_distance / 1e6)
}

/// Batch calculation of density corrected volumes for multiple z_max values
fn calculate_v_dc_max_batch<F: Fn(f64) -> f64 + Sync>(
    max_zs: &[f64],
    delta_z: F,
    cosmo: &Cosmology,
) -> Vec<f64> {
    if max_zs.is_empty() {
        return vec![];
    }

    let max_limit = max_zs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let xs = linspace(0., max_limit, 1000);
    let bin_size = xs[1] - xs[0];

    let ys: Vec<f64> = xs
        .par_iter()
        .map(|&z| delta_z(z) * cosmo.differential_comoving_distance(z))
        .collect();

    let mut cumsum = Vec::with_capacity(ys.len());
    let mut sum = 0.0;
    for &y in &ys {
        sum += y * bin_size;
        cumsum.push(sum);
    }

    max_zs
        .par_iter()
        .map(|&limit| {
            if limit <= 0.0 {
                return 0.0;
            }
            if limit >= max_limit {
                return cumsum.last().unwrap() * 4.0 * PI;
            }

            let idx = ((limit / max_limit) * (xs.len() - 1) as f64) as usize;
            let idx = idx.min(cumsum.len() - 1);

            let v_dc = if idx < cumsum.len() - 1 {
                let x0 = xs[idx];
                let x1 = xs[idx + 1];
                let y0 = cumsum[idx];
                let y1 = cumsum[idx + 1];
                y0 + (y1 - y0) * (limit - x0) / (x1 - x0)
            } else {
                cumsum[idx]
            };

            v_dc * 4.0 * PI
        })
        .collect()
}

fn _reflect_into_range(value: f64, min_value: f64, max_value: f64) -> f64 {
    let mut v = value;
    // Keep reflecting until within bounds
    loop {
        if v < min_value {
            v = 2.0 * min_value - v;
        } else if v > max_value {
            v = 2.0 * max_value - v;
        } else {
            return v;
        }
        // Safety check to avoid infinite loops
        if (v - value).abs() > 10.0 * (max_value - min_value) {
            return value.max(min_value).min(max_value);
        }
    }
}

fn interp_many(x_vals: &[f64], y_vals: &[f64], queries: &[f64]) -> Vec<f64> {
    assert_eq!(
        x_vals.len(),
        y_vals.len(),
        "x_vals and y_vals length mismatch"
    );

    let mut indexed: Vec<(usize, f64)> = queries.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut results = vec![0.0; queries.len()];
    let mut i = 1;

    for (orig_idx, qx) in indexed {
        while i < x_vals.len() && x_vals[i] < qx {
            i += 1;
        }

        let val = if i == 0 {
            y_vals[0]
        } else if i >= x_vals.len() {
            y_vals[y_vals.len() - 1]
        } else {
            let x0 = x_vals[i - 1];
            let x1 = x_vals[i];
            let y0 = y_vals[i - 1];
            let y1 = y_vals[i];
            y0 + (y1 - y0) * (qx - x0) / (x1 - x0)
        };

        results[orig_idx] = val;
    }

    results
}

/// Sample from truncated normal distribution using statrs for CDF/inverse CDF
/// Uses parallel sampling with thread-local RNGs for performance
fn truncated_normal(mu: f64, sigma: f64, a: f64, b: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }

    let normal = StatrsNormal::new(mu, sigma).unwrap();
    let pa = normal.cdf(a);
    let pb = normal.cdf(b);

    (0..n)
        .into_par_iter()
        .map_init(
            || SmallRng::from_rng(&mut rand::rng()),
            |rng, _| {
                let u = rng.random_range(pa..pb);
                normal.inverse_cdf(u)
            },
        )
        .collect()
}

/// Optimized populate_volume using proper truncated normal distribution
fn populate_volume(
    z: f64,
    z_max: f64,
    n_points: f64,
    covol_z_vals: &[f64],
    covol_vals: &[f64],
) -> Vec<f64> {
    let n_target = n_points as usize;
    if n_target == 0 {
        return Vec::new();
    }

    let volume = interp(covol_z_vals, covol_vals, z, &InterpMode::Extrapolate);
    let max_volume = interp(covol_z_vals, covol_vals, z_max, &InterpMode::Extrapolate);

    let sigma_vol = 0.33 * max_volume; //3.5e9;

    // Truncate at ±2σ
    let lower = (volume - 2.0 * sigma_vol).max(0.0);
    let upper = (volume + 2.0 * sigma_vol).min(max_volume);

    // Handle edge cases
    if upper <= lower {
        let v = lower.max(0.0).min(max_volume);
        let zs = vec![v; n_target];
        return interp_many(covol_vals, covol_z_vals, &zs);
    }

    // Use truncated normal sampling
    let volumes = truncated_normal(volume, sigma_vol, lower, upper, n_target);

    // Convert volumes to redshifts in one vectorized pass
    interp_many(covol_vals, covol_z_vals, &volumes)
}

/// OPTIMIZED: Takes counts directly instead of computing histogram
fn approximate_delta_y_from_counts(n_g: &[i32], n_r: &[i32], n_clone: f64) -> Vec<f64> {
    n_g.iter()
        .zip(n_r.iter())
        .map(|(&g, &r)| {
            let r_safe = if r == 0 { 1 } else { r };
            n_clone * (g as f64 / r_safe as f64)
        })
        .collect()
}

fn approximate_delta_x(redshift_bins: &[f64]) -> Vec<f64> {
    redshift_bins
        .windows(2)
        .map(|b| (b[0] + b[1]) / 2.)
        .collect()
}

/// Main function which generates the randoms vector of the survey.
pub fn generate_randoms(
    redshifts: Vec<f64>,
    mags: Vec<f64>,
    z_lim: f64,
    maglim: f64,
    n_clone: i32,
    iterations: i32,
    cosmo: Cosmology,
) -> Vec<f64> {
    let bin_width = calculate_fd(redshifts.clone());
    let redshift_bins = arange(0., z_lim, bin_width);

    // Pre-compute x_values once (doesn't change)
    let x_values = approximate_delta_x(&redshift_bins);

    // Pre-compute n_g once (doesn't change)
    let n_g = histogram(redshifts.clone(), redshift_bins.clone());

    let max_zs: Vec<f64> = redshifts
        .par_iter()
        .zip(mags.par_iter())
        .map(|(&z, &m)| calculate_max_z(m, z, maglim, &cosmo))
        .collect();

    // PRE-COMPUTE comoving volume lookup table ONCE
    let z_vals = linspace(0., z_lim, 1000);
    let covol: Vec<f64> = z_vals
        .par_iter()
        .map(|&z| cosmo.comoving_volume(z))
        .collect();

    let mut randoms: Vec<f64> = redshifts
        .par_iter()
        .zip(max_zs.par_iter())
        .flat_map(|(&z, &max_z)| populate_volume(z, max_z, n_clone as f64, &z_vals, &covol))
        .collect();

    let v_maxes: Vec<f64> = interp_many(&z_vals, &covol, &max_zs);
    let n_clone_f64 = n_clone as f64;

    for iteration in 0..iterations {
        println!("Iteration {}/{}", iteration + 1, iterations);

        // Compute histogram of randoms (only thing that changes)
        // OPTIMIZATION: Pass reference to avoid clone if histogram supports it
        let n_r = histogram(randoms.clone(), redshift_bins.clone());

        // Use pre-computed n_g and new n_r
        let y_values = approximate_delta_y_from_counts(&n_g, &n_r, n_clone_f64);

        let delta_func = |z| interp(&x_values, &y_values, z, &InterpMode::Constant(1.));

        let v_dc_maxes = calculate_v_dc_max_batch(&max_zs, delta_func, &cosmo);

        let n_new: Vec<f64> = v_maxes
            .par_iter()
            .zip(v_dc_maxes.par_iter())
            .map(|(&v, &v_dc)| n_clone_f64 * v / v_dc)
            .collect();

        randoms = redshifts
            .par_iter()
            .zip(max_zs.par_iter())
            .zip(n_new.par_iter())
            .flat_map(|((&z, &z_max), &n)| populate_volume(z, z_max, n, &z_vals, &covol))
            .collect();
    }

    // Filter out any randoms that are exactly at z_lim to avoid boundary pileup
    let epsilon = 1e-10;
    randoms.retain(|&z| z < z_lim - epsilon);

    randoms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_z() {
        let mag = 19.8;
        let mag_lim = 19.8;
        let z = 0.2;
        let cosmo = Cosmology {
            omega_m: 0.3,
            omega_k: 0.,
            omega_l: 0.7,
            h0: 100.,
        };
        assert!((calculate_max_z(mag, z, mag_lim, &cosmo) - z).abs() < 1e-7);

        let mag = 19.6;
        assert!((calculate_max_z(mag, z, mag_lim, &cosmo) - 0.21713688).abs() < 1e-5);
    }

    #[test]
    fn test_inverse_interp_binary_exact_match() {
        let x_vals = [0.0, 1.0, 2.0, 3.0];
        let y_vals = [0.0, 10.0, 20.0, 30.0];
        let target = [20.];
        let result = interp_many(&y_vals, &x_vals, &target);
        assert!((result[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_inverse_interp_binary_interpolation() {
        let x_vals = [0.0, 1.0, 2.0, 3.0];
        let y_vals = [0.0, 10.0, 20.0, 30.0];
        let result = interp_many(&y_vals, &x_vals, &[15.0]);
        assert!((result[0] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_inverse_interp_binary_below_range() {
        let x_vals = [0.0, 1.0, 2.0, 3.0];
        let y_vals = [0.0, 10.0, 20.0, 30.0];
        let result = interp_many(&y_vals, &x_vals, &[-5.0]);
        dbg!("{:?}", result.clone());
        assert!((result[0] + 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_inverse_interp_binary_above_range() {
        let x_vals = [0.0, 1.0, 2.0, 3.0];
        let y_vals = [0.0, 10.0, 20.0, 30.0];
        let result = interp_many(&y_vals, &x_vals, &[35.0]);
        assert!((result[0] - 3.0).abs() < 1e-12);
    }
}
