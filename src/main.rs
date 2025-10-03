use std::fs::File;
use std::path::Path;
use std::time::Instant;

use polars::prelude::*;
use csv::Writer;
use randoms::cosmology::Cosmology;
use randoms::generate_randoms;


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

    let now = Instant::now();
    let randoms = generate_randoms(redshifts, mags, max_z, maglim, n_clone, iterations, cosmo);
    println!("Total Time Taken: {:?}", now.elapsed());



    // writing the randoms
    let file = File::create("delete_randoms.csv").unwrap();
    let mut wtr = Writer::from_writer(file);
    for x in randoms.iter() {
        wtr.write_record(&[x.to_string()]).unwrap()
    }
    Ok(())
}
