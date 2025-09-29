pub mod constants;
pub mod cosmology;
pub mod histogram;

use integrate::adaptive_quadrature;
use libm::log10;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;
use polars::prelude::*;

use crate::cosmology::Cosmology;
use crate::histogram::{arange, calculate_fd, histogram};

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

fn approximate_delta_x(redshift_bins: Vec<f64>) -> Vec<f64> {
    redshift_bins
        .windows(2)
        .map(|b| (b[0] + b[1]) / 2.)
        .collect()
}

fn read_gama<P: AsRef<Path>>(file_path: P) -> PolarsResult<DataFrame> {
    CsvReader::from_path(file_path)?
        .with_delimiter(b' ')
        .has_header(true)
        .finish()
}

fn main() -> PolarsResult<()> {
    let maglim = 19.8;
    let n_clone = 400.;
    let file_name = "/Users/00115372/Desktop/prototype_nz/g09_galaxies.dat";
    let cosmo = Cosmology {
            omega_m: 0.3,
            omega_k: 0.,
            omega_l: 0.7,
            h0: 70.,
        };
    
    //read in file
    let df = read_gama(file_name)?;
    let redshifts = df.column("z")?
        .f64()?
        .into_no_null_iter()
        .collect::<Vec<f64>>();

    let mags = df.column("Rpetro")?
        .f64()?
        .into_no_null_iter()
        .collect::<Vec<f64>>();
    
    let bin_width = calculate_fd(redshifts);
    let max_z = 1.; //TODO: How do we fix this thing.

    let max_zs = redshifts
        .iter()
        .zip(mags)
        .map(|(&z, m)| calculate_max_z(mag, z, maglim, cosmo))
        .collect::<Vec<f64>>();

    let first_randoms = redshifts
        .iter()
        .zip(max_zs)
        .map(|(&z, max_z)| populate_volume(z, max_z, n_points, cosmo))
        .collect::<Vec<Vec<f64>>>();

    

}
