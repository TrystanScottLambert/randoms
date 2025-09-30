pub mod constants;
pub mod cosmology;
pub mod histogram;

use csv::Writer;
use interp::{InterpMode, interp};
use libm::log10;
use polars::prelude::*;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::f64::consts::PI;
use std::fs::File;
use std::path::Path;

use crate::cosmology::Cosmology;
use crate::histogram::{arange, calculate_fd, histogram};

/// calcuates the redshift at which the current galaxy with magntidue mag and redshift z would be
/// visible
fn calculate_max_z(mag: f64, z: f64, mag_lim: f64, cosmo: &Cosmology) -> f64 {
    let distance_measured = cosmo.luminosity_distance(z) * 1e6; // Mpc to pc
    let max_distance = 10_f64.powf((mag_lim - mag + 5. * log10(distance_measured)) / 5.);
    cosmo.inverse_lumdist(max_distance / 1e6) // back to Mpc
}

fn fast_rough_integral<F: Fn(f64) -> f64 + Sync>(function: F, limit: f64) -> f64 {
    let bin_size = 0.01;
    let xs = arange(0., limit, bin_size);
    let ys: Vec<f64> = xs.iter().map(|&b| function(b)).collect();
    ys.into_iter().sum::<f64>() * bin_size
}

/// Calculates the density corrected volume based on the given overdensity function delta_z
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

/// Create randoms values within
fn populate_volume(z: f64, z_max: f64, n_points: f64, cosmo: &Cosmology) -> Vec<f64> {
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

    let z_vals = arange(0., 1., 0.001);
    let covol: Vec<f64> = z_vals.iter().map(|&z| cosmo.comoving_volume(z)).collect();

    random_volumes
        .iter()
        .map(|&v| inverse_interp_binary(&covol, &z_vals, v))
        .collect()
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
    let file = File::open(file_path)?;
    let parse_opts = CsvParseOptions::default().with_separator(b' ');
    let options = CsvReadOptions::default()
        .with_parse_options(parse_opts)
        .with_has_header(true);
    let df = CsvReader::new(file).with_options(options).finish()?;
    Ok(df)
}

fn generate_randoms(
    redshifts: Vec<f64>,
    mags: Vec<f64>,
    max_z: f64,
    maglim: f64,
    n_clone: i32,
    iterations: i32,
    cosmo: Cosmology,
) -> Vec<f64> {
    let bin_width = calculate_fd(redshifts.clone());
    let redshift_bins = arange(0., max_z, bin_width);
    let max_zs = redshifts
        .clone()
        .iter()
        .zip(mags)
        .map(|(&z, m)| calculate_max_z(m, z, maglim, &cosmo))
        .collect::<Vec<f64>>();

    let mut randoms: Vec<f64> = redshifts
        .par_iter()
        .zip(max_zs.clone())
        .flat_map(|(&z, max_z)| populate_volume(z, max_z, n_clone as f64, &cosmo))
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
            .flat_map(|((&z, z_max), n)| populate_volume(z, z_max, n, &cosmo))
            .collect();
    }

    randoms
}
fn main() -> PolarsResult<()> {
    let maglim = 19.8;
    let n_clone = 400;
    let iterations = 10;
    let max_z = 1.;

    let file_name = "/Users/00115372/Desktop/prototype_nz/g09_galaxies.dat";
    let cosmo = Cosmology {
        omega_m: 0.3,
        omega_k: 0.,
        omega_l: 0.7,
        h0: 70.,
    };

    //read in file
    let df = read_gama(file_name)?;
    let redshifts = df
        .column("Z")?
        .f64()?
        .into_no_null_iter()
        .collect::<Vec<f64>>();

    let mags = df
        .column("Rpetro")?
        .f64()?
        .into_no_null_iter()
        .collect::<Vec<f64>>();

    let randoms = generate_randoms(redshifts, mags, max_z, maglim, n_clone, iterations, cosmo);

    // writing the randoms
    let file = File::create("delete_randoms.csv").unwrap();
    let mut wtr = Writer::from_writer(file);
    for x in randoms.iter() {
        wtr.write_record(&[x.to_string()]).unwrap()
    }
    Ok(())
}
