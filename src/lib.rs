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
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::f64::consts::PI;

use crate::cosmology::Cosmology;
use crate::histogram::{arange, calculate_fd, histogram, linspace};

/// Calcuates the redshift at which the current galaxy with magntidue mag and redshift z would be visible.
///
/// Calculates the luminosity distance at the given redshift, z and then uses the distance modulus
/// to solve for the distance at the mag_lim. The inverse luminosity distance is then used to calculate
/// the max distance.
fn calculate_max_z(mag: f64, z: f64, mag_lim: f64, cosmo: &Cosmology) -> f64 {
    let distance_measured = cosmo.luminosity_distance(z) * 1e6; // Mpc to pc
    let max_distance = 10_f64.powf((mag_lim - mag + 5. * log10(distance_measured)) / 5.);
    cosmo.inverse_lumdist(max_distance / 1e6) // back to Mpc
}

/// A rough rieman sum numerical approximation for fast but less accurate numerical integration.
///
/// This creates an x grid consiting of 1000 points, equally spaced from 0, to the limit of
/// integration and then evaluates the function at these points. The Rieman sum is then performed
/// to calcualte the intergal.
fn fast_rough_integral<F: Fn(f64) -> f64 + Sync>(function: F, limit: f64) -> f64 {
    let xs = linspace(0., limit, 1000);
    let bin_size = xs[1] - xs[0];
    let ys: Vec<f64> = xs.iter().map(|&b| function(b)).collect();
    ys.into_iter().sum::<f64>() * bin_size
}

/// Calculates the density corrected volume based on the given overdensity function delta_z
///
/// Evaluates the the product of the over density function (delta_z) and the differential comovoing
/// distance to get the density-weighted maximum comoving volume (Eq. 5 Farrow+2015).
fn calculate_v_dc_max<F: Fn(f64) -> f64 + Sync>(z_max: f64, delta_z: F, cosmo: &Cosmology) -> f64 {
    let integrand = |z: f64| delta_z(z) * cosmo.differential_comoving_distance(z);
    let v_dc = fast_rough_integral(integrand, z_max);
    v_dc * 4. * PI
}

/// Reflect the value if it is outside of the min values back within the bounds.
fn _reflect_into_range(value: f64, min_value: f64, max_value: f64) -> f64 {
    if value < min_value {
        2. * min_value - value
    } else if value > max_value {
        2. * max_value - value
    } else {
        value
    }
}

/// Helper function to speed up interpolation by performing a binary search for the target value.
fn inverse_interp_binary(y_vals: &[f64], x_vals: &[f64], target_y: f64) -> f64 {
    // Binary search to find bracketing indices
    let idx = match y_vals.binary_search_by(|probe| probe.partial_cmp(&target_y).unwrap()) {
        Ok(i) => return x_vals[i], // Exact match
        Err(i) => i,
    };

    if idx == 0 {
        return x_vals[0];
    }
    if idx >= y_vals.len() {
        return x_vals[x_vals.len() - 1];
    }

    // Linear interpolation between idx-1 and idx
    let x0 = x_vals[idx - 1];
    let x1 = x_vals[idx];
    let y0 = y_vals[idx - 1];
    let y1 = y_vals[idx];

    x0 + (x1 - x0) * (target_y - y0) / (y1 - y0)
}

/// Create randoms redshift values, popuated within the volume.
///
/// Randomly generates volumes from a normal distribution with a mean of the comoving volume defined
/// at z and a static sigma of 3.5e9. Volumes less than 0 and more than the max volume defined by
/// z_max are ignored.
fn populate_volume(z: f64, z_max: f64, n_points: f64, z_limit: f64, cosmo: &Cosmology) -> Vec<f64> {
    let mut random_volumes = Vec::new();
    let volume = cosmo.comoving_volume(z);
    let max_volume = cosmo.comoving_volume(z_max);
    let sigma_vol = 3.5e9;
    let mut min_vol = volume - 2. * sigma_vol;
    let mut max_vol = volume + 2. * sigma_vol;
    if min_vol < 0. {
        min_vol = 0.;
    }
    if max_vol > max_volume {
        max_vol = max_volume;
    }
    let normal = Normal::new(volume, sigma_vol).unwrap();
    let mut counter = 0.;
    while counter < n_points {
        let v = normal.sample(&mut rand::rng());
        if (v < max_vol) && (v > min_vol) {
            random_volumes.push(v);
            counter += 1.;
        }
    }

    let z_vals = linspace(0., z_limit, 1000);
    let covol: Vec<f64> = z_vals.iter().map(|&z| cosmo.comoving_volume(z)).collect();

    random_volumes
        .iter()
        .map(|&v| inverse_interp_binary(&covol, &z_vals, v))
        .collect()
}

/// Helper function to calculate the y values of the overdensity function.
///
/// Histograms the real and randoms data and returns the ratio of the two multiplied by the given
/// n_clone.
fn approximate_delta_y(
    real_redshifts: Vec<f64>,
    random_redshifts: Vec<f64>,
    redshift_bins: Vec<f64>,
    n_clone: f64,
) -> Vec<f64> {
    let n_g = histogram(real_redshifts, redshift_bins.clone());
    let n_r = histogram(random_redshifts, redshift_bins.clone());
    let n_r = n_r
        .iter()
        .map(|&c| if c == 0 { 1 } else { c })
        .collect::<Vec<i32>>();
    n_g.iter()
        .zip(n_r)
        .map(|(&g, r)| n_clone * (g as f64 / r as f64))
        .collect::<Vec<f64>>()
}

/// Helper function that determine the x values of the approximate delta z function.
///
/// Calcualtes the midpoints of the redshift bins.
fn approximate_delta_x(redshift_bins: Vec<f64>) -> Vec<f64> {
    redshift_bins
        .windows(2)
        .map(|b| (b[0] + b[1]) / 2.)
        .collect()
}

/// Main function which generates the randoms vector of the survey.
/// 
/// # Arguments
/// 
/// * `redshifts` The redshifts of the survey.
/// * `mags` The apparent magnitudes of the galaxies in the survey.
/// * `z_lim` The redshift up to which the user wishes to evaluate. This can just be a reasonble choice greater than the max redshift.
/// * `maglim` The magnitude limit of the survey.
/// * `n_clone` The number of times more the randoms catalogue will be than the original catalogue.
/// * `iterations` The number of iterations to iteratively solve for delta_z (usually 5-10 is plenty).
/// * `cosmo` The Cosmology object from the Cosmology module.
///
/// This function iteratively solves for the overdensity function by first cloning all galxies
/// n_clone number of times, and calcualte the delta_z function using that randoms catalogue.
/// See section 3.1.1 of Farrow+2015 for more details.
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
    let max_zs = redshifts
        .clone()
        .iter()
        .zip(mags)
        .map(|(&z, m)| calculate_max_z(m, z, maglim, &cosmo))
        .collect::<Vec<f64>>();

    let mut randoms: Vec<f64> = redshifts
        .par_iter()
        .zip(max_zs.clone())
        .flat_map(|(&z, max_z)| populate_volume(z, max_z, n_clone as f64, z_lim, &cosmo))
        .collect();

    let v_maxes = max_zs
        .clone()
        .iter()
        .map(|&z| cosmo.comoving_volume(z))
        .collect::<Vec<f64>>();

    for _ in 0..iterations {
        let x_values = approximate_delta_x(redshift_bins.clone());
        let y_values = approximate_delta_y(
            redshifts.clone(),
            randoms.clone(),
            redshift_bins.clone(),
            n_clone as f64,
        );
        let delta_func = |z| interp(&x_values, &y_values, z, &InterpMode::Constant(1.));
        let v_dc_maxes: Vec<f64> = max_zs
            .par_iter()
            .map(|&z| calculate_v_dc_max(z, delta_func, &cosmo))
            .collect();
        let n_new: Vec<f64> = v_maxes
            .iter()
            .zip(v_dc_maxes)
            .map(|(&v, v_dc)| n_clone as f64 * v / v_dc)
            .collect();

        randoms = redshifts
            .clone()
            .par_iter()
            .zip(max_zs.clone())
            .zip(n_new)
            .flat_map(|((&z, z_max), n)| populate_volume(z, z_max, n, z_lim, &cosmo))
            .collect();
    }

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
    fn test_fast_rough_integral_with_sin() {
        let limit = std::f64::consts::PI;
        let result = fast_rough_integral(|x| x.sin(), limit);
        // The true integral is 2.0, allow a small tolerance:
        let expected = 2.0;
        let tolerance = 1e-3; // since it’s a rough method
        assert!((result - expected).abs() < tolerance);
    }

    #[test]
    fn test_inverse_interp_binary_exact_match() {
        let x_vals = [0.0, 1.0, 2.0, 3.0];
        let y_vals = [0.0, 10.0, 20.0, 30.0];

        // Exact y = 20 should give x = 2.0
        let result = inverse_interp_binary(&y_vals, &x_vals, 20.0);
        assert!((result - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_inverse_interp_binary_interpolation() {
        let x_vals = [0.0, 1.0, 2.0, 3.0];
        let y_vals = [0.0, 10.0, 20.0, 30.0];

        // y = 15 lies between 10 and 20, expect x = 1.5
        let result = inverse_interp_binary(&y_vals, &x_vals, 15.0);
        assert!((result - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_inverse_interp_binary_below_range() {
        let x_vals = [0.0, 1.0, 2.0, 3.0];
        let y_vals = [0.0, 10.0, 20.0, 30.0];

        // y < min(y_vals) should return first x
        let result = inverse_interp_binary(&y_vals, &x_vals, -5.0);
        assert!((result - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_inverse_interp_binary_above_range() {
        let x_vals = [0.0, 1.0, 2.0, 3.0];
        let y_vals = [0.0, 10.0, 20.0, 30.0];

        // y > max(y_vals) should return last x
        let result = inverse_interp_binary(&y_vals, &x_vals, 35.0);
        assert!((result - 3.0).abs() < 1e-12);
    }
}
