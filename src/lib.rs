//! # randoms
//!
//! Generate random (unclustered) catalogues for magnitude-limited redshift surveys, using the
//! density-corrected V_max cloning method of Cole (2011), optionally with the Gaussian
//! "windowed clones" modification of Farrow et al. (2015, §3.1.2).
//!
//! Outline (Cole 2011 §2-4, Farrow 2015 §3.1):
//! 1. For each galaxy compute its absolute magnitude (with k+e corrections) and the redshift
//!    range [z_min,i, z_max,i] over which it would satisfy the survey magnitude limits.
//! 2. Each galaxy distributes its clones over that range in P-weighted volume, either uniformly
//!    (Cole) or with a reflected Gaussian window around its own position (Farrow). This gives,
//!    for each galaxy, the fraction f_ip of its clones expected in each redshift bin p.
//! 3. Iterate the radial overdensity Delta_p:
//!    w_i = 1 / <Delta>_i with <Delta>_i = sum_p f_ip Delta_p (= V_dc,max / V_max for the
//!    uniform window), normalised so <w> = 1 (Cole's normalisation constraint, with mu = 0);
//!    expected random counts n_r,p = n_clone sum_i w_i f_ip; Delta_p = n_clone n_g,p / n_r,p
//!    (Farrow eq. 7), optionally shrunk towards 1 with the Cole eq. 18 J3 prior.
//! 4. Clone each galaxy n_clone * w_i times (stochastic rounding, Cole footnote 4) and place
//!    each clone by drawing from its window, recomputing the apparent magnitude at the new z.
//!
//! The expected n_r is computed analytically (as in the Fortran `estimate_nofz`) rather than
//! from Monte Carlo randoms each iteration, so the iteration is noise-free and cheap; the
//! Monte Carlo catalogue is drawn once at the end.

pub mod config;
pub mod constants;
pub mod cosmology;
pub mod model;
pub mod tables;
pub mod window;

use std::fmt;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

pub use crate::model::{DensityEvolution, ECorrection, KCorrection, Model};
pub use crate::tables::DistanceTable;
pub use crate::window::{Sigma, Window};
use crate::window::{NormalQuadrature, fold, truncated_standard_normal};

/// Survey selection.
#[derive(Debug, Clone)]
pub struct Survey {
    pub zmin: f64,
    pub zmax: f64,
    /// Faint apparent-magnitude limit.
    pub mag_faint: f64,
    /// Optional bright apparent-magnitude limit.
    pub mag_bright: Option<f64>,
    /// Survey solid angle in steradians. Only affects the absolute volume units (and so the
    /// meaning of `Sigma::Volume` and `four_pi_j3`).
    pub solid_angle_sr: f64,
}

/// Algorithm settings.
#[derive(Debug, Clone)]
pub struct RandomsConfig {
    /// Target ratio of randoms to galaxies.
    pub n_clone: f64,
    /// Number of uniform redshift bins between survey zmin and zmax used for Delta(z).
    pub nzbin: usize,
    pub max_iter: usize,
    /// Convergence when max |Delta_new - Delta_old| < tol.
    pub tol: f64,
    pub seed: u64,
    pub window: Window,
    /// Optional 4 pi J3 (Cole 2011 eq. 18) clustering prior on Delta, in the volume units of the
    /// table ((Mpc/h)^3 for h0 = 100). `None` = raw estimate (Farrow 2015 eq. 7).
    pub four_pi_j3: Option<f64>,
    /// Number of points in the redshift lookup table.
    pub table_points: usize,
    /// Number of quadrature points across the Gaussian window.
    pub window_quadrature_points: usize,
}

impl Default for RandomsConfig {
    fn default() -> Self {
        RandomsConfig {
            n_clone: 400.,
            nzbin: 40,
            max_iter: 1000,
            tol: 1e-4,
            seed: 42,
            window: Window::Uniform,
            four_pi_j3: None,
            table_points: 20001,
            window_quadrature_points: 400,
        }
    }
}

/// Derived per-galaxy quantities.
#[derive(Debug, Clone, Copy)]
pub struct GalaxyProps {
    pub absmag: f64,
    /// Redshift range over which the galaxy satisfies the selection.
    pub zmin: f64,
    pub zmax: f64,
    /// P-weighted accessible volume, V_max.
    pub vmax: f64,
    /// Final clone weight w_i (mean 1); expected clones = n_clone * w_i.
    pub weight: f64,
}

/// The random catalogue (struct-of-arrays).
#[derive(Debug, Clone, Default)]
pub struct Randoms {
    pub z: Vec<f64>,
    /// Apparent magnitude of the clone at its new redshift.
    pub mag: Vec<f64>,
    /// Index of the parent galaxy in the input arrays.
    pub parent: Vec<usize>,
}

impl Randoms {
    pub fn len(&self) -> usize {
        self.z.len()
    }
    pub fn is_empty(&self) -> bool {
        self.z.is_empty()
    }
}

/// The radial overdensity solution.
#[derive(Debug, Clone)]
pub struct DeltaBins {
    /// nzbin + 1 bin edges in redshift.
    pub z_edges: Vec<f64>,
    /// Galaxies per bin.
    pub n_gal: Vec<f64>,
    /// Expected randoms per bin divided by n_clone (directly comparable to n_gal).
    pub n_ran: Vec<f64>,
    pub delta: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct RandomsResult {
    pub randoms: Randoms,
    pub galaxies: Vec<GalaxyProps>,
    pub bins: DeltaBins,
    /// max |Delta change| for each iteration.
    pub delta_change: Vec<f64>,
    pub converged: bool,
}

#[derive(Debug, Clone)]
pub struct RandomsError(pub String);

impl fmt::Display for RandomsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for RandomsError {}

fn err<T>(msg: impl Into<String>) -> Result<T, RandomsError> {
    Err(RandomsError(msg.into()))
}

/// Sparse per-galaxy bin fractions, CSR layout.
struct BinFractions {
    offsets: Vec<usize>,
    bins: Vec<u32>,
    fracs: Vec<f64>,
}

impl BinFractions {
    fn row(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let (a, b) = (self.offsets[i], self.offsets[i + 1]);
        self.bins[a..b]
            .iter()
            .zip(&self.fracs[a..b])
            .map(|(&p, &f)| (p as usize, f))
    }
}

fn bin_of(edges: &[f64], x: f64) -> usize {
    let nb = edges.len() - 1;
    edges
        .partition_point(|&e| e <= x)
        .saturating_sub(1)
        .min(nb - 1)
}

/// Per-galaxy geometry in P-weighted volume.
#[derive(Clone, Copy)]
struct Geometry {
    absmag: f64,
    zlo: f64,
    zhi: f64,
    ulo: f64,
    uhi: f64,
    ucen: f64,
}

fn window_fractions(
    g: &Geometry,
    window: &ResolvedWindow,
    u_edges: &[f64],
    quad: Option<&NormalQuadrature>,
) -> Vec<(u32, f64)> {
    let len = g.uhi - g.ulo;
    if len <= 0. {
        return vec![(bin_of(u_edges, g.ucen) as u32, 1.)];
    }
    let p0 = bin_of(u_edges, g.ulo);
    let p1 = bin_of(u_edges, g.uhi);
    match window {
        ResolvedWindow::Uniform => (p0..=p1)
            .filter_map(|p| {
                let ov = g.uhi.min(u_edges[p + 1]) - g.ulo.max(u_edges[p]);
                (ov > 0.).then_some((p as u32, ov / len))
            })
            .collect(),
        ResolvedWindow::Gaussian { sigma, .. } => {
            let quad = quad.expect("quadrature for gaussian window");
            let mut acc = vec![0.; p1 - p0 + 1];
            for (&t, &w) in quad.nodes.iter().zip(&quad.weights) {
                let x = fold(g.ucen + sigma * t, g.ulo, g.uhi);
                acc[bin_of(u_edges, x).clamp(p0, p1) - p0] += w;
            }
            acc.into_iter()
                .enumerate()
                .filter(|&(_, f)| f > 0.)
                .map(|(k, f)| ((p0 + k) as u32, f))
                .collect()
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ResolvedWindow {
    Uniform,
    Gaussian { sigma: f64, truncate: f64 },
}

impl ResolvedWindow {
    fn sample<R: Rng>(&self, g: &Geometry, rng: &mut R) -> f64 {
        match *self {
            ResolvedWindow::Uniform => g.ulo + (g.uhi - g.ulo) * rng.random::<f64>(),
            ResolvedWindow::Gaussian { sigma, truncate } => {
                let t = truncated_standard_normal(truncate, rng);
                fold(g.ucen + sigma * t, g.ulo, g.uhi)
            }
        }
    }
}

/// Solve the Cole quadratic 0 = -delta_raw + delta + f delta (delta - 1) (Fortran
/// `estimate_delta`), which shrinks delta towards 1 where the expected fluctuations are large.
fn shrink_delta(delta_raw: f64, f: f64) -> f64 {
    if f > 5e-4 {
        (f - 1. + ((1. - f).powi(2) + 4. * f * delta_raw).sqrt()) / (2. * f)
    } else {
        delta_raw + f * (delta_raw - delta_raw * delta_raw)
    }
}

/// Generate a random catalogue.
///
/// `z` and `mag` are the redshifts and apparent magnitudes (in the selection band) of the
/// galaxies. Every galaxy must lie within the survey selection (`zmin <= z <= zmax`,
/// `mag_bright <= mag <= mag_faint`); filter beforehand.
pub fn generate_randoms(
    z: &[f64],
    mag: &[f64],
    survey: &Survey,
    model: &Model,
    cfg: &RandomsConfig,
) -> Result<RandomsResult, RandomsError> {
    // ---- validation -------------------------------------------------------------------
    if z.len() != mag.len() {
        return err("z and mag have different lengths");
    }
    if z.is_empty() {
        return err("empty catalogue");
    }
    if !(survey.zmax > survey.zmin && survey.zmin >= 0.) {
        return err("need 0 <= survey.zmin < survey.zmax");
    }
    if cfg.nzbin == 0 || cfg.n_clone <= 0. || cfg.table_points < 2 {
        return err("nzbin, n_clone and table_points must be positive");
    }
    let eps = 1e-9;
    let bad = z
        .iter()
        .zip(mag)
        .filter(|&(&zi, &mi)| {
            !(zi.is_finite() && mi.is_finite())
                || zi < survey.zmin
                || zi > survey.zmax
                || mi > survey.mag_faint + eps
                || survey.mag_bright.is_some_and(|b| mi < b - eps)
        })
        .count();
    if bad > 0 {
        return err(format!(
            "{bad} galaxies are outside the survey selection (z in [{}, {}], mag limits) or \
             non-finite; filter them before calling generate_randoms",
            survey.zmin, survey.zmax
        ));
    }

    // ---- tables and bins -----------------------------------------------------------------
    let table = DistanceTable::new(model, survey.zmax, survey.solid_angle_sr, cfg.table_points);
    let nb = cfg.nzbin;
    let z_edges: Vec<f64> = (0..=nb)
        .map(|p| survey.zmin + (survey.zmax - survey.zmin) * p as f64 / nb as f64)
        .collect();
    let u_edges: Vec<f64> = z_edges.iter().map(|&ze| table.pvolume_at(ze)).collect();
    let survey_u = u_edges[nb] - u_edges[0];

    let window = match cfg.window {
        Window::Uniform => ResolvedWindow::Uniform,
        Window::Gaussian { sigma, truncate } => {
            let sigma = match sigma {
                Sigma::Volume(v) => v,
                Sigma::FractionOfSurvey(f) => f * survey_u,
            };
            if !(sigma > 0. && truncate > 0.) {
                return err("gaussian window needs sigma > 0 and truncate > 0");
            }
            ResolvedWindow::Gaussian { sigma, truncate }
        }
    };
    let quad = match window {
        ResolvedWindow::Gaussian { truncate, .. } => Some(NormalQuadrature::new(
            truncate,
            cfg.window_quadrature_points.max(16),
        )),
        ResolvedWindow::Uniform => None,
    };

    // ---- per-galaxy geometry ----------------------------------------------------------
    let u_survey_min = u_edges[0];
    let geom: Vec<Geometry> = z
        .par_iter()
        .zip(mag.par_iter())
        .map(|(&zi, &mi)| {
            let absmag = mi - table.mag_offset_at(zi);
            // clamp so the galaxy's own redshift is always inside its range (guards against
            // interpolation round-off for galaxies right at the limit)
            let zhi = table.limit_z(absmag, survey.mag_faint).min(survey.zmax).max(zi);
            let zlo = survey
                .mag_bright
                .map_or(0., |b| table.limit_z(absmag, b))
                .max(survey.zmin)
                .min(zi);
            let ulo = table.pvolume_at(zlo).max(u_survey_min);
            let uhi = table.pvolume_at(zhi);
            let ucen = table.pvolume_at(zi).clamp(ulo, uhi);
            Geometry {
                absmag,
                zlo,
                zhi,
                ulo,
                uhi,
                ucen,
            }
        })
        .collect();

    let mut n_gal = vec![0.; nb];
    for &zi in z {
        n_gal[bin_of(&z_edges, zi)] += 1.;
    }

    // ---- bin fractions (geometry only, independent of Delta) ----------------------------
    let rows: Vec<Vec<(u32, f64)>> = geom
        .par_iter()
        .map(|g| window_fractions(g, &window, &u_edges, quad.as_ref()))
        .collect();
    let mut fr = BinFractions {
        offsets: Vec::with_capacity(rows.len() + 1),
        bins: Vec::new(),
        fracs: Vec::new(),
    };
    fr.offsets.push(0);
    for row in rows {
        for (p, f) in row {
            fr.bins.push(p);
            fr.fracs.push(f);
        }
        fr.offsets.push(fr.bins.len());
    }

    // ---- iterate Delta ------------------------------------------------------------------
    let ngal = z.len();
    let bin_volume: Vec<f64> = (0..nb)
        .map(|p| table.volume_at(z_edges[p + 1]) - table.volume_at(z_edges[p]))
        .collect();

    let weights_for = |delta: &[f64]| -> Vec<f64> {
        let mut w: Vec<f64> = (0..ngal)
            .into_par_iter()
            .map(|i| {
                let md: f64 = fr.row(i).map(|(p, f)| f * delta[p]).sum();
                if md > 0. { 1. / md } else { 1. }
            })
            .collect();
        let mean = w.iter().sum::<f64>() / ngal as f64;
        w.iter_mut().for_each(|x| *x /= mean);
        w
    };
    let expected_nran = |w: &[f64]| -> Vec<f64> {
        // data-normalised: sum = ngal
        (0..ngal)
            .into_par_iter()
            .fold(
                || vec![0.; nb],
                |mut acc, i| {
                    for (p, f) in fr.row(i) {
                        acc[p] += w[i] * f;
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.; nb],
                |mut a, b| {
                    a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                    a
                },
            )
    };

    let mut delta = vec![1.; nb];
    let mut delta_change = Vec::new();
    let mut converged = false;
    for _ in 0..cfg.max_iter {
        let w = weights_for(&delta);
        let n_ran = expected_nran(&w);
        let new: Vec<f64> = (0..nb)
            .map(|p| {
                if n_ran[p] <= 0. {
                    return 1.;
                }
                let raw = n_gal[p] / n_ran[p];
                match cfg.four_pi_j3 {
                    None => raw,
                    Some(j3) => {
                        // f = 1 / (sigma^2 n_ran), sigma^2 = (1 + n_ran J3 / V) / n_ran
                        let f = 1. / (1. + n_ran[p] * j3 / bin_volume[p]);
                        shrink_delta(raw, f)
                    }
                }
            })
            .collect();
        let change = new
            .iter()
            .zip(&delta)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        delta = new;
        delta_change.push(change);
        if change < cfg.tol {
            converged = true;
            break;
        }
    }

    let w = weights_for(&delta);
    let n_ran = expected_nran(&w);

    // ---- generate the clones ------------------------------------------------------------
    let clones: Vec<Vec<(f64, f64)>> = (0..ngal)
        .into_par_iter()
        .map(|i| {
            let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
            rng.set_stream(i as u64);
            let target = cfg.n_clone * w[i];
            let base = target.floor();
            let nrep = base as usize + usize::from(rng.random::<f64>() < target - base);
            let g = &geom[i];
            (0..nrep)
                .map(|_| {
                    let u = window.sample(g, &mut rng);
                    let zc = table.z_at_pvolume(u).clamp(g.zlo, g.zhi);
                    (zc, g.absmag + table.mag_offset_at(zc))
                })
                .collect()
        })
        .collect();

    let total: usize = clones.iter().map(Vec::len).sum();
    let mut randoms = Randoms {
        z: Vec::with_capacity(total),
        mag: Vec::with_capacity(total),
        parent: Vec::with_capacity(total),
    };
    for (i, c) in clones.into_iter().enumerate() {
        for (zc, mc) in c {
            randoms.z.push(zc);
            randoms.mag.push(mc);
            randoms.parent.push(i);
        }
    }

    let galaxies = geom
        .iter()
        .zip(&w)
        .map(|(g, &wi)| GalaxyProps {
            absmag: g.absmag,
            zmin: g.zlo,
            zmax: g.zhi,
            vmax: g.uhi - g.ulo,
            weight: wi,
        })
        .collect();

    Ok(RandomsResult {
        randoms,
        galaxies,
        bins: DeltaBins {
            z_edges,
            n_gal,
            n_ran,
            delta,
        },
        delta_change,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosmology::Cosmology;

    fn model() -> Model {
        Model::new(Cosmology {
            omega_m: 0.3,
            omega_k: 0.,
            omega_l: 0.7,
            h0: 100.,
        })
    }

    fn survey() -> Survey {
        Survey {
            zmin: 0.01,
            zmax: 0.4,
            mag_faint: 19.0,
            mag_bright: None,
            solid_angle_sr: 0.05,
        }
    }

    /// Unclustered mock: galaxies uniform in volume with a Gaussian LF, optionally with a
    /// redshift shell boosted by `boost` (a fake overdensity).
    fn mock(n_target: usize, boost: Option<(f64, f64, f64)>, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let s = survey();
        let m = model();
        let t = DistanceTable::new(&m, s.zmax, s.solid_angle_sr, 20001);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let (u0, u1) = (t.pvolume_at(s.zmin), t.pvolume_at(s.zmax));
        let (mut z, mut mag) = (vec![], vec![]);
        while z.len() < n_target {
            let zi = t.z_at_pvolume(u0 + (u1 - u0) * rng.random::<f64>());
            let copies = match boost {
                Some((a, b, f)) if zi > a && zi < b => {
                    f.floor() as usize + usize::from(rng.random::<f64>() < f.fract())
                }
                _ => 1,
            };
            for _ in 0..copies {
                let absmag: f64 = -20.5 + 1.2 * rng.sample::<f64, _>(rand_distr::StandardNormal);
                let mi = absmag + t.mag_offset_at(zi);
                if mi <= s.mag_faint {
                    z.push(zi);
                    mag.push(mi);
                }
            }
        }
        (z, mag)
    }

    #[test]
    fn rejects_galaxies_outside_selection() {
        let r = generate_randoms(
            &[0.1, 0.5],
            &[18., 18.],
            &survey(),
            &model(),
            &RandomsConfig::default(),
        );
        assert!(r.is_err());
    }

    #[test]
    fn unclustered_mock_gives_flat_delta() {
        let (z, mag) = mock(40_000, None, 3);
        let cfg = RandomsConfig {
            n_clone: 20.,
            nzbin: 10,
            ..Default::default()
        };
        let r = generate_randoms(&z, &mag, &survey(), &model(), &cfg).unwrap();
        assert!(r.converged);
        for (p, d) in r.bins.delta.iter().enumerate() {
            // within 4 sigma of Poisson noise
            let tol = 4. / r.bins.n_gal[p].sqrt();
            assert!((d - 1.).abs() < tol, "bin {p}: delta {d} (n_gal {})", r.bins.n_gal[p]);
        }
        let expected = 20. * z.len() as f64;
        assert!((r.randoms.len() as f64 / expected - 1.).abs() < 0.01);
        // every random satisfies the selection
        for (&zc, &mc) in r.randoms.z.iter().zip(&r.randoms.mag) {
            assert!((0.01..=0.4).contains(&zc));
            assert!(mc <= 19.0 + 1e-6, "clone mag {mc}");
        }
    }

    #[test]
    fn recovers_injected_overdensity() {
        // Double the density in 0.15 < z < 0.2; the randoms should not follow it.
        let (z, mag) = mock(60_000, Some((0.15, 0.2, 2.0)), 5);
        let cfg = RandomsConfig {
            n_clone: 20.,
            nzbin: 39, // bin width 0.01 so the shell is exactly 5 bins
            ..Default::default()
        };
        let r = generate_randoms(&z, &mag, &survey(), &model(), &cfg).unwrap();
        let b = &r.bins;
        let mut inside = vec![];
        let mut outside = vec![];
        for p in 0..b.delta.len() {
            let zc = 0.5 * (b.z_edges[p] + b.z_edges[p + 1]);
            if zc > 0.15 && zc < 0.2 {
                inside.push(b.delta[p]);
            } else if b.n_gal[p] > 500. {
                outside.push(b.delta[p]);
            }
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let ratio = mean(&inside) / mean(&outside);
        assert!((ratio - 2.).abs() < 0.15, "overdensity ratio {ratio}");
    }

    #[test]
    fn seeded_runs_are_reproducible() {
        let (z, mag) = mock(3_000, None, 9);
        let cfg = RandomsConfig {
            n_clone: 5.,
            nzbin: 8,
            window: Window::Gaussian {
                sigma: Sigma::FractionOfSurvey(0.05),
                truncate: 2.,
            },
            ..Default::default()
        };
        let a = generate_randoms(&z, &mag, &survey(), &model(), &cfg).unwrap();
        let b = generate_randoms(&z, &mag, &survey(), &model(), &cfg).unwrap();
        assert_eq!(a.randoms.z, b.randoms.z);
        assert_eq!(a.randoms.parent, b.randoms.parent);
    }

    #[test]
    fn stochastic_rounding_preserves_mean() {
        let (z, mag) = mock(20_000, None, 11);
        let cfg = RandomsConfig {
            n_clone: 2.5,
            nzbin: 10,
            ..Default::default()
        };
        let r = generate_randoms(&z, &mag, &survey(), &model(), &cfg).unwrap();
        let expected: f64 = r.galaxies.iter().map(|g| 2.5 * g.weight).sum();
        let got = r.randoms.len() as f64;
        // Bernoulli variance <= n/4
        assert!((got - expected).abs() < 5. * (z.len() as f64 / 4.).sqrt());
    }

    #[test]
    fn gaussian_window_keeps_clones_near_parent() {
        let (z, mag) = mock(10_000, None, 13);
        let cfg = RandomsConfig {
            n_clone: 10.,
            nzbin: 10,
            window: Window::Gaussian {
                sigma: Sigma::FractionOfSurvey(0.02),
                truncate: 2.,
            },
            ..Default::default()
        };
        let r = generate_randoms(&z, &mag, &survey(), &model(), &cfg).unwrap();
        let mean_abs_dz: f64 = r
            .randoms
            .z
            .iter()
            .zip(&r.randoms.parent)
            .map(|(&zc, &p)| (zc - z[p]).abs())
            .sum::<f64>()
            / r.randoms.len() as f64;
        assert!(mean_abs_dz < 0.03, "mean |dz| {mean_abs_dz}");
        // and Delta still ~1 for an unclustered mock
        for (d, n) in r.bins.delta.iter().zip(&r.bins.n_gal) {
            assert!((d - 1.).abs() < 4. / n.sqrt(), "{:?}", r.bins.delta);
        }
    }
}
