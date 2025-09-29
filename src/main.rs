pub mod constants;
pub mod cosmology;
pub mod histogram;

use csv::Writer;
use integrate::adaptive_quadrature;
use interp::{InterpMode, interp};
use libm::log10;
use polars::prelude::*;
use rand_distr::{Distribution, Normal};
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
    let bin_size = 0.0001;
    let xs = arange(0., limit, bin_size);
    let ys: Vec<f64> = xs.iter().map(|&b| function(b)).collect();
    ys.into_iter().sum::<f64>() * bin_size
}

/// Calculates the density corrected volume based on the given overdensity function delta_z
fn calculate_v_dc_max<F: Fn(f64) -> f64 + Sync>(z_max: f64, delta_z: F, cosmo: &Cosmology) -> f64 {
    let integrand = |z: f64| delta_z(z) * cosmo.differential_comoving_distance(z);
    // testing
    let redshifts = arange(0., 1., 0.01);
    let integrands: Vec<f64> = redshifts.iter().map(|&z| integrand(z)).collect();
    let file = File::create("delete_integral.csv").unwrap();
    let mut wtr = Writer::from_writer(file);
    for (x, y) in redshifts.iter().zip(integrands.iter()) {
        wtr.write_record(&[x.to_string(), y.to_string()]).unwrap();
    }
    wtr.flush().unwrap();
    //
    //let tolerance = 1e-5;
    //let min_h = 1e-7;
    let v_dc = fast_rough_integral(integrand, z_max);
    // println!("Zmax thing: {}", z_max);
    // let v_dc =
    //     adaptive_quadrature::adaptive_simpson_method(integrand, 1e-3, z_max, min_h, tolerance)
    //         .unwrap();
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
    let file = File::open(file_path)?;
    let parse_opts = CsvParseOptions::default().with_separator(b' ');
    let options = CsvReadOptions::default()
        .with_parse_options(parse_opts)
        .with_has_header(true);
    let df = CsvReader::new(file).with_options(options).finish()?;
    Ok(df)
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
    println!("Here");
    let bin_width = calculate_fd(redshifts.clone());

    let max_z = 1.; //TODO: How do we fix this thing.
    let redshift_bins = arange(0., max_z, bin_width);
    println!("bins done");
    let max_zs = redshifts
        .clone()
        .iter()
        .zip(mags)
        .map(|(&z, m)| calculate_max_z(m, z, maglim, &cosmo))
        .collect::<Vec<f64>>();
    println!("Max Zs done");
    //println!("{:?}", max_zs);
    let randoms: Vec<f64> = redshifts
        .iter()
        .zip(max_zs.clone())
        .flat_map(|(&z, max_z)| populate_volume(z, max_z, n_clone, &cosmo))
        .collect();
    println!("randoms done");
    let v_maxes = max_zs
        .clone()
        .iter()
        .map(|&z| cosmo.comoving_volume(z))
        .collect::<Vec<f64>>();
    println!("v_maxes done");
    let mut counter = 1;
    while counter < 15 {
        println!("counter");
        let x_values = approximate_delta_x(redshift_bins.clone());
        let y_values = approximate_delta_y(
            redshifts.clone(),
            randoms.clone(),
            redshift_bins.clone(),
            n_clone,
        );
        let delta_func = |z| interp(&x_values, &y_values, z, &InterpMode::Constant(1.));
        // testing
        let x = arange(0.01, 1., 0.01);
        let y: Vec<f64> = x.iter().map(|&r| delta_func(r)).collect();
        let file = File::create("delete.csv")?;
        let mut wtr = Writer::from_writer(file);
        for (_x, _y) in x.iter().zip(y.iter()) {
            wtr.write_record(&[_x.to_string(), _y.to_string()]).unwrap();
        }
        wtr.flush().unwrap();
        //
        let v_dc_maxes: Vec<f64> = max_zs
            .iter()
            .map(|&z| calculate_v_dc_max(z, delta_func, &cosmo))
            .collect();
        let n_new: Vec<f64> = v_maxes
            .iter()
            .zip(v_dc_maxes)
            .map(|(&v, v_dc)| n_clone * v / v_dc)
            .collect();
        let randoms: Vec<f64> = redshifts
            .clone()
            .iter()
            .zip(max_zs.clone())
            .zip(n_new)
            .flat_map(|((&z, z_max), n)| populate_volume(z, z_max, n, &cosmo))
            .collect();
        let new_ratio = randoms.len() / redshifts.len();
        println!("{new_ratio}");
        counter += 1;
    }
    Ok(())
}
