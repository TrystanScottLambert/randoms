pub mod constants;
pub mod cosmology;
pub mod histogram;

use integrate::adaptive_quadrature;
use libm::log10;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

use crate::cosmology::Cosmology;

/// calcuates the redshift at which the current galaxy with magntidue mag and redshift z would be
/// visible
fn calculate_max_z(mag: f64, z: f64, mag_lim: f64, cosmo: Cosmology) -> f64 {
    let distance_measured = cosmo.luminosity_distance(z) * 1e6; // Mpc to pc
    let max_distance = 10_f64.powf((mag_lim - mag + 5. * log10(distance_measured)) / 5.);
    cosmo.inverse_lumdist(max_distance)
}

/// Calculates the density corrected volume based on the given overdensity function delta_z
fn calculate_v_dc_max<F: Fn(f64) -> f64 + Sync + Copy>(
    z_max: f64,
    delta_z: F,
    cosmo: Cosmology,
) -> f64 {
    let integrand = |z: f64| delta_z(z) * cosmo.differential_comoving_distance(z);
    let tolerance = 1e-5;
    let min_h = 1e-7;
    let v_dc =
        adaptive_quadrature::adaptive_simpson_method(integrand, 0.0, z_max, min_h, tolerance)
            .expect("Value too close to zero. Must be within 10e-8");
    v_dc * 4. * PI
}

/// Reflect the value if it is outside of the min values back within the bounds.
fn reflect_into_range(value: f64, min_value: f64, max_value: f64) -> f64 {
    if value < min_value {
        2. * min_value - value
    } else if value > max_value {
        2. * max_value - value
    } else {
        value
    }
}

/// Create randoms values within
fn populate_volume(z: f64, z_max: f64, n_points: i32, cosmo: Cosmology) -> Vec<f64> {
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
    let mut counter = 0;

    while counter < n_points {
        let v = normal.sample(&mut rand::rng());
        if (v < max_vol) && (v > min_vol) {
            random_volumes.push(v);
            counter += 1;
        }
    }
    random_volumes
        .iter()
        .map(|&v| cosmo.inverse_covol(v))
        .collect::<Vec<f64>>()
}

fn approximate_delta(
    real_redshifts: Vec<f64>,
    random_redshifts: Vec<f64>,
    redshift_bins: Vec<f64>,
    n_clone: i32,
) {
    let center_redshift_bins: Vec<f64> = redshift_bins
        .windows(2)
        .map(|b| (b[0] + b[1]) / 2.)
        .collect();

}

fn main() {
    println!("Hello, world!");
}
