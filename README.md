# randoms

Generate random (unclustered) catalogues for magnitude-limited redshift surveys using the
density-corrected V_max cloning method of [Cole (2011)](https://arxiv.org/abs/1104.0009), with
the optional Gaussian "windowed clones" of Farrow et al. (2015, MNRAS 454, 2120, §3.1.2).

Each galaxy is cloned `n_clone * w_i` times, with `w_i = V_max / V_dc,max` (or its windowed
equivalent), and each clone is placed at a random position within the volume where the galaxy
would still pass the survey magnitude limits. The radial overdensity Δ(z) is iterated to
convergence, so large-scale structure in the data is not imprinted on the randoms.

## Usage

```sh
cargo run --release -- configs/gama_dr4.toml
```

The TOML config specifies:
- the input CSV (redshift column, a magnitude column or a flux column with zero-point, and
  extra columns to copy to the clones);
- the survey limits (zmin, zmax, faint and optional bright magnitude limit, solid angle);
- cosmology, the k-correction polynomial, e(z) = -q z, and P(z) = exp(a z);
- `n_clone`, `nzbin`, convergence settings, `seed`, and the window:
  - `type = "none"`: Cole (2011), uniform in V_max;
  - `type = "gaussian"` with `sigma_volume` or `sigma_fraction`: Farrow (2015).

Outputs:
- the randoms (`z, mag, absmag, parent_row, <carried columns>`);
- Δ(z) per bin;
- a per-galaxy table (`absmag, zmin, zmax, vmax, weight`).

Angular positions are not generated. Use `parent_row` or the carried RA/Dec columns, or draw
positions from your own angular mask.

## Library

```rust
let result = randoms::generate_randoms(&z, &mag, &survey, &model, &config)?;
```

## Notes

- Volumes are P-weighted comoving volumes within the survey solid angle. With h0 = 100 they are
  in (Mpc/h)^3, which is the unit `sigma_volume` uses (Farrow: 3.5e6 for GAMA II).
- The expected random n(z) at each iteration is computed analytically, as in the Fortran
  `estimate_nofz`. The Monte Carlo catalogue is drawn once at the end, with stochastic rounding
  of clone counts. Runs are reproducible for a given `seed`.
- Δ(z) and P(z) are degenerate (Cole 2011 §3). A Δ that trends with redshift usually means the
  evolution model (q, a) does not fit the data.
