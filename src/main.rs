//! Command-line driver: `randoms <config.toml>`.

use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use randoms::config::Config;
use randoms::generate_randoms;

struct Catalogue {
    z: Vec<f64>,
    mag: Vec<f64>,
    carry: Vec<Vec<String>>,
}

fn column(headers: &csv::StringRecord, name: &str) -> Result<usize, String> {
    headers
        .iter()
        .position(|h| h.trim() == name)
        .ok_or_else(|| format!("column '{name}' not found; available: {headers:?}"))
}

fn read_catalogue(cfg: &Config) -> Result<Catalogue, Box<dyn Error>> {
    let inp = &cfg.input;
    let survey = cfg.survey();
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(inp.delimiter as u8)
        .trim(csv::Trim::All)
        .from_path(&inp.path)
        .map_err(|e| format!("opening {}: {e}", inp.path.display()))?;
    let headers = rdr.headers()?.clone();
    let iz = column(&headers, &inp.redshift)?;
    let imag = match (&inp.magnitude, &inp.flux) {
        (Some(m), None) => column(&headers, m)?,
        (None, Some(f)) => column(&headers, &f.column)?,
        _ => unreachable!("validated in Config::from_file"),
    };
    let icarry: Vec<usize> = inp
        .carry
        .iter()
        .map(|c| column(&headers, c))
        .collect::<Result<_, _>>()?;
    let iselect: Vec<(usize, f64, f64)> = inp
        .select
        .iter()
        .map(|s| column(&headers, &s.column).map(|i| (i, s.min, s.max)))
        .collect::<Result<_, _>>()?;

    let (mut total, mut bad, mut zcut, mut magcut, mut unselected) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut cat = Catalogue {
        z: vec![],
        mag: vec![],
        carry: vec![],
    };
    for rec in rdr.records() {
        let rec = rec?;
        total += 1;
        let selected = iselect.iter().all(|&(i, lo, hi)| {
            rec[i].parse::<f64>().is_ok_and(|v| v >= lo && v <= hi)
        });
        if !selected {
            unselected += 1;
            continue;
        }
        let z: f64 = rec[iz].parse().unwrap_or(f64::NAN);
        let raw: f64 = rec[imag].parse().unwrap_or(f64::NAN);
        let mag = match &inp.flux {
            Some(f) => {
                if raw > 0. {
                    f.zero_point - 2.5 * raw.log10()
                } else {
                    f64::NAN
                }
            }
            None => raw,
        };
        if !(z.is_finite() && mag.is_finite()) {
            bad += 1;
            continue;
        }
        if z < survey.zmin || z > survey.zmax {
            zcut += 1;
            continue;
        }
        if mag > survey.mag_faint || survey.mag_bright.is_some_and(|b| mag < b) {
            magcut += 1;
            continue;
        }
        cat.z.push(z);
        cat.mag.push(mag);
        cat.carry
            .push(icarry.iter().map(|&i| rec[i].to_string()).collect());
    }
    println!(
        "read {total} rows from {}: kept {}, dropped {unselected} outside `select` ranges, {bad} unparseable/non-positive flux, \
         {zcut} outside {} <= z <= {}, {magcut} outside magnitude limits",
        inp.path.display(),
        cat.z.len(),
        survey.zmin,
        survey.zmax
    );
    Ok(cat)
}

fn create(path: &Path) -> Result<BufWriter<File>, Box<dyn Error>> {
    Ok(BufWriter::new(File::create(path).map_err(|e| {
        format!("creating {}: {e}", path.display())
    })?))
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn run(config_path: PathBuf) -> Result<(), Box<dyn Error>> {
    let cfg = Config::from_file(&config_path)?;
    let survey = cfg.survey();
    let model = cfg.model();
    let rcfg = cfg.randoms_config()?;

    let cat = read_catalogue(&cfg)?;

    let now = Instant::now();
    let res = generate_randoms(&cat.z, &cat.mag, &survey, &model, &rcfg)?;
    println!(
        "{} after {} iterations (final max |dDelta| = {:.2e}); {} randoms ({:.2} x galaxies) in {:.2?}",
        if res.converged { "converged" } else { "NOT converged" },
        res.delta_change.len(),
        res.delta_change.last().copied().unwrap_or(f64::NAN),
        res.randoms.len(),
        res.randoms.len() as f64 / cat.z.len() as f64,
        now.elapsed()
    );

    let now = Instant::now();
    let mut w = create(&cfg.output.randoms)?;
    let carry_header: String = cfg.input.carry.iter().map(|c| format!(",{c}")).collect();
    writeln!(w, "z,mag,absmag,parent_row{carry_header}")?;
    let carry_lines: Vec<String> = cat
        .carry
        .iter()
        .map(|row| row.iter().map(|v| format!(",{}", csv_field(v))).collect())
        .collect();
    let r = &res.randoms;
    for k in 0..r.len() {
        let p = r.parent[k];
        writeln!(
            w,
            "{:.6},{:.4},{:.4},{}{}",
            r.z[k], r.mag[k], res.galaxies[p].absmag, p, carry_lines[p]
        )?;
    }
    w.flush()?;
    println!(
        "wrote {} in {:.2?}",
        cfg.output.randoms.display(),
        now.elapsed()
    );

    if let Some(path) = &cfg.output.delta {
        let b = &res.bins;
        let mut w = create(path)?;
        writeln!(w, "z_lo,z_hi,z_mid,n_gal,n_ran_scaled,delta")?;
        for p in 0..b.delta.len() {
            let (lo, hi) = (b.z_edges[p], b.z_edges[p + 1]);
            writeln!(
                w,
                "{lo:.5},{hi:.5},{:.5},{},{:.3},{:.6}",
                0.5 * (lo + hi),
                b.n_gal[p],
                b.n_ran[p],
                b.delta[p]
            )?;
        }
        w.flush()?;
        println!("wrote {}", path.display());
    }

    if let Some(path) = &cfg.output.galaxies {
        let mut w = create(path)?;
        writeln!(w, "row,z,mag,absmag,zmin,zmax,vmax,weight{carry_header}")?;
        for (i, g) in res.galaxies.iter().enumerate() {
            writeln!(
                w,
                "{i},{:.6},{:.4},{:.4},{:.6},{:.6},{:.6e},{:.6}{}",
                cat.z[i], cat.mag[i], g.absmag, g.zmin, g.zmax, g.vmax, g.weight, carry_lines[i]
            )?;
        }
        w.flush()?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: randoms <config.toml>   (see configs/gama_dr4.toml)");
        std::process::exit(2);
    };
    if let Err(e) = run(PathBuf::from(path)) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
