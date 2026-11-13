//! TOML configuration for the `randoms` command-line tool. See `configs/gama_dr4.toml`.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::cosmology::Cosmology;
use crate::model::{DensityEvolution, ECorrection, KCorrection, Model};
use crate::window::{Sigma, Window};
use crate::{RandomsConfig, Survey};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub input: InputConfig,
    pub output: OutputConfig,
    pub survey: SurveyConfig,
    pub cosmology: CosmologyConfig,
    #[serde(default)]
    pub kcorrection: KCorrection,
    #[serde(default)]
    pub ecorrection: ECorrection,
    #[serde(default)]
    pub density_evolution: DensityEvolution,
    /// Reference redshift for k+e = 0 and P = 1.
    #[serde(default)]
    pub zref: f64,
    #[serde(default)]
    pub randoms: RandomsSection,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputConfig {
    pub path: PathBuf,
    #[serde(default = "default_delimiter")]
    pub delimiter: char,
    /// Redshift column.
    pub redshift: String,
    /// Apparent magnitude column (give this or `flux`).
    pub magnitude: Option<String>,
    /// Flux column converted with mag = zero_point - 2.5 log10(flux) (e.g. 8.9 for Jy -> AB).
    pub flux: Option<FluxColumn>,
    /// Extra columns copied from each parent galaxy to its clones.
    #[serde(default)]
    pub carry: Vec<String>,
    /// Keep only rows with min <= column <= max for every entry (e.g. one survey field).
    #[serde(default)]
    pub select: Vec<RangeSelect>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RangeSelect {
    pub column: String,
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluxColumn {
    pub column: String,
    pub zero_point: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputConfig {
    pub randoms: PathBuf,
    /// Optional Delta(z) table.
    pub delta: Option<PathBuf>,
    /// Optional per-galaxy table (absmag, zmin, zmax, vmax, weight).
    pub galaxies: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurveyConfig {
    pub zmin: f64,
    pub zmax: f64,
    pub mag_faint: f64,
    pub mag_bright: Option<f64>,
    pub solid_angle_deg2: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CosmologyConfig {
    pub omega_m: f64,
    pub omega_l: f64,
    pub h0: f64,
    /// Defaults to 1 - omega_m - omega_l.
    pub omega_k: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RandomsSection {
    pub n_clone: f64,
    pub nzbin: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub seed: u64,
    pub four_pi_j3: Option<f64>,
    pub table_points: usize,
    pub window: WindowConfig,
}

impl Default for RandomsSection {
    fn default() -> Self {
        let d = RandomsConfig::default();
        RandomsSection {
            n_clone: d.n_clone,
            nzbin: d.nzbin,
            max_iter: d.max_iter,
            tol: d.tol,
            seed: d.seed,
            four_pi_j3: d.four_pi_j3,
            table_points: d.table_points,
            window: WindowConfig::None,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum WindowConfig {
    /// Cole (2011): uniform within V_max.
    #[default]
    None,
    /// Farrow et al. (2015): Gaussian in volume, give exactly one of `sigma_volume` or
    /// `sigma_fraction`.
    Gaussian {
        sigma_volume: Option<f64>,
        sigma_fraction: Option<f64>,
        #[serde(default = "default_truncate")]
        truncate: f64,
    },
}

fn default_delimiter() -> char {
    ','
}

fn default_truncate() -> f64 {
    2.0
}

impl Config {
    /// Read a config file. Relative paths inside it are resolved against the config file's
    /// directory.
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("reading config {}: {e}", path.display()))?;
        let mut cfg: Config = toml::from_str(&text)?;
        let base = path.parent().unwrap_or(Path::new("."));
        let fix = |p: &mut PathBuf| {
            if p.is_relative() {
                *p = base.join(&*p);
            }
        };
        fix(&mut cfg.input.path);
        fix(&mut cfg.output.randoms);
        if let Some(p) = cfg.output.delta.as_mut() {
            fix(p);
        }
        if let Some(p) = cfg.output.galaxies.as_mut() {
            fix(p);
        }
        if cfg.input.magnitude.is_some() == cfg.input.flux.is_some() {
            return Err("give exactly one of input.magnitude or input.flux".into());
        }
        Ok(cfg)
    }

    pub fn survey(&self) -> Survey {
        let s = &self.survey;
        Survey {
            zmin: s.zmin,
            zmax: s.zmax,
            mag_faint: s.mag_faint,
            mag_bright: s.mag_bright,
            solid_angle_sr: s.solid_angle_deg2 * (std::f64::consts::PI / 180.).powi(2),
        }
    }

    pub fn model(&self) -> Model {
        let c = &self.cosmology;
        Model {
            cosmology: Cosmology {
                omega_m: c.omega_m,
                omega_l: c.omega_l,
                omega_k: c.omega_k.unwrap_or(1. - c.omega_m - c.omega_l),
                h0: c.h0,
            },
            kcorr: self.kcorrection.clone(),
            ecorr: self.ecorrection.clone(),
            density: self.density_evolution.clone(),
            zref: self.zref,
        }
    }

    pub fn randoms_config(&self) -> Result<RandomsConfig, String> {
        let r = &self.randoms;
        let window = match r.window {
            WindowConfig::None => Window::Uniform,
            WindowConfig::Gaussian {
                sigma_volume,
                sigma_fraction,
                truncate,
            } => {
                let sigma = match (sigma_volume, sigma_fraction) {
                    (Some(v), None) => Sigma::Volume(v),
                    (None, Some(f)) => Sigma::FractionOfSurvey(f),
                    _ => {
                        return Err(
                            "gaussian window: give exactly one of sigma_volume or sigma_fraction"
                                .into(),
                        );
                    }
                };
                Window::Gaussian { sigma, truncate }
            }
        };
        Ok(RandomsConfig {
            n_clone: r.n_clone,
            nzbin: r.nzbin,
            max_iter: r.max_iter,
            tol: r.tol,
            seed: r.seed,
            window,
            four_pi_j3: r.four_pi_j3,
            table_points: r.table_points,
            ..RandomsConfig::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gama_config_parses() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("configs/gama_dr4.toml");
        let cfg = Config::from_file(&path).unwrap();
        let rc = cfg.randoms_config().unwrap();
        assert!(matches!(rc.window, Window::Gaussian { .. }));
        let m = cfg.model();
        assert_eq!(m.cosmology.omega_k, 0.);
        assert!((m.ecorr.e(1.) + 1.75).abs() < 1e-12);
        assert!((cfg.survey().solid_angle_sr - 240. * (std::f64::consts::PI / 180.).powi(2)).abs() < 1e-12);
    }
}
