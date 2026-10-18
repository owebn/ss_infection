# ss_infection

One-dimensional reaction–diffusion simulations (FEniCSx) and dispersion-relation
theory for phage infection fronts spreading through a bacterial lawn. The
models cover an Erlang-distributed latent period, abortive-infection (Abi)
cells, a diffusible interferon-like death signal that primes neighbouring
hosts, phage that read that signal to change latency, burst or lysogeny, and
temperate phage with a prophage bank.

Notation follows the draft manuscript *Self-generated immunity halos and phage
counter-strategies in a spreading plaque* (Wingreen group): V free phage, S
susceptible cells, R₁…Rₘ the refractory (maturation) chain with Rₘ primed,
F the signal, β adsorption.

## Installation

PDE simulations need FEniCSx (DOLFINx 0.9), PETSc and MPI, which are easiest to
get from conda-forge:

```bash
conda env create -f environment.yml
```

```bash
conda activate ss-infection
```

This installs the package in editable mode with `flake8` for development.
Theory-only use (`--theoretical`) needs just NumPy, SciPy and Matplotlib, so
`pip install -e .` in any Python ≥ 3.10 environment is enough for that.

## Quick start

Run one simulation and write every output to a folder:

```bash
python -m ss_infection --model delay_chain --n_stages 4 --tau 0.25 --L 100 --t_end 30 --output_dir runs/delay_chain
```

Leading-edge wave speed from the dispersion relation (no PDE):

```bash
python -m ss_infection --model lysogeny_delay_chain --theoretical --n_stages 4 --tau 0.25 --p_lys 0.3
```

Parameter sweep from a JSON file (one CSV row per run, run in parallel):

```bash
python -m ss_infection --sweep examples/lysogeny_sweep.json -o runs/lysogeny.csv -j 4
```

Re-draw every figure from a saved run without simulating, optionally with new
regime thresholds:

```bash
python -m ss_infection --from_history runs/delay_chain/history.npz --output_dir runs/delay_chain --stall_frac 0.2
```

The manuscript's stop-and-go parameter set B:

```bash
python -m ss_infection --model immune_signaling --n_stages 1 --tau 0.5 --M 40 --K 1 --alpha_priming 20 --n_priming 3 --mu 2 --D_s 40 --gamma_s 0.5 --b_R 0.5 --rho_reversion 0.05 --p_ext 1e-4 --L 300 --t_end 200 --dt 0.05 --history_every 20 --output_dir runs/setB
```

`python -m ss_infection --help` lists every option. After installation the
same command is also available as `ss-infection`.

## Models

| `--model` | Description | Manuscript |
|---|---|---|
| `delay_chain` | Phage V, susceptible cells S, latent chain I₁…Iₙ (rate w = n/τ, burst b) | §1.1 |
| `abi` | Lawn with a fraction `f_abi` of abortive cells R (burst 0) | §2.1 |
| `abi_heterogeneous` | Naive region of length `L_naive` followed by an abortive patch of length `L_primed` | §2.4 |
| `lysogeny_delay_chain` | Temperate phage: constant lysogeny probability `p_lys` over a decision window; lysogens L induce at rate ν | §6.1 (no signal) |
| `immune_signaling` | Death signal F (yield M, diffusion D_s, decay γ_s) primes S → R₁…Rₘ above threshold K; maturation races lysis, so infected cells keep maturing (I⁽ʲ⁾ₖ) | §§3.1, 3.5 |
| `immune_signaling_evasion` | As above; phage switch latency (`--tau_slow`) and burst (`--b_slow` or `--burst_mode frontier`) when F exceeds the trigger F_p | §4.5 |
| `immune_signaling_lysogeny_evasion` | As above; phage switch lysogeny probability from `p_lys` to `p_lys_hi` when F exceeds F_p | §6 |

Strategy parameters default to the non-sensing values (`tau_slow = tau`,
`b_slow = b`, `p_lys_hi = p_lys`, `F_p = K`), so an unset run is the
non-adaptive control.

### Parameter names

Each parameter has a canonical name (used in CSVs and history files) and
manuscript-notation aliases accepted on the command line, in sweep JSON and in
`simulate(params)`:

| Symbol | Canonical | Aliases |
|---|---|---|
| β, β_I, β_L | `f_B_val`, `f_I_val`, `f_L_val` | `beta`, `beta_I`, `beta_L` |
| S₀ | `B0_val` | `S0` |
| b, b_R | `b_val`, `b_abi` | `b`, `b_R` |
| D, D_s | `D_val`, `D_signal` | `D`, `D_s` |
| γ_s, K, M | `gamma_signal`, `T_signal`, `signal_yield` | `gamma_s`, `K`, `M` |
| α, μ, h, m | `k_priming`, `mu_priming`, `hill_signal`, `n_priming` | `alpha_priming`, `mu`, `h`*, `m`* |
| F_p, h_p | `sigma_p`, `hill_p` | `F_p`, `h_p` |
| δ_V, V_ext | `delta_P`, `p_ext` | `delta_V`, `V_ext` |

\* sweep JSON and Python only (no one-letter command-line flags). `alpha` is
reserved as an older alias of `f_abi`.

## Outputs

### Single run (`--output_dir DIR`)

| File | Contents |
|---|---|
| `history.npz` | Cached run: profiles every `--history_every` steps, per-step series, metrics, parameters. Written as `history.npz.tmp` while running, so a killed run leaves a readable partial file |
| `kymograph.png` | Uninfected density in space–time with model overlays (c* fit, patch boundaries, primed edge) |
| `speed.png` | Smoothed tip speed, reference speed c_ref, stalls shaded red; signal models add L_halo on a second axis |
| `phage.png` | Total free phage ∫V dx over time (log scale), stalls shaded |
| `snapshot.png` | Final spatial profiles |
| `metrics.json`, `params.json` | Run metrics (including the regime and stall intervals) and the full parameter set |

Load a saved run in Python with
`ss_infection.simulation.SimulationResult.from_history(path)`. The figure
functions in `ss_infection.plotting` take that object.

### Sweeps (`--sweep config.json -o results.csv`)

A sweep JSON has `sweep` (ranges with `start`, `end` and `num` or `step`;
`"scale": "log"` optional), `constants`, and `adaptive` (domain size and
resolution chosen from a Fisher–KPP estimate). See `examples/`. Each run's
history goes to `results_histories/run_XXXXX.npz`, linked from the CSV's
`history_file` column (disable with `--no_histories`).

CSV columns depend on the model:

* every run: `c_star`, `extinction`, `p_end`, `t_extinction`
* PDE runs: regime metrics `regime`, `c_ref`, `c_mean`, `c_cv`, `X_max`,
  `t_arrest`, `n_stalls`, `T_cycle`, `L_step`, `hit_domain_end`
* signal models: `L_halo_max/final`, `primed_band_width_max/final`,
  `sigma_tip_final` (edge signal F_f), `z_K_final`, `warning_median`

### Front regimes

`regime` is one of `extinct`, `silent`, `stop_and_go`, `stall`,
`single_stall`, `wave`, `wave_fast`, `wave_halo`, `unsteady`, `undetermined`
or `failed`. It is computed from a sliding-window fit of the leading
infection edge: a stall is a stretch where the tip speed stays below
`stall_frac × c_ref` (default 0.25) for at least `stall_min`. All thresholds
can be set with `--regime_transient`, `--regime_window`, `--stall_frac`,
`--stall_min`, `--cv_steady` and `--halo_tol`. See `regimes.py`.

## Adding a model

Subclass `BaseModel` in `models.py`, implement `num_components`,
`create_weak_formulation` and `profile_groups`, set `name` and
`initial_condition`, and add the class to `MODEL_CLASSES`. The command line,
sweeps, history files, CSV columns and plots pick it up from the class's
declarations (`trackers`, `kymograph_overlays`, `speed_panel_series`,
`theoretical_speed`, `param_aliases`, …), which are documented in
`BaseModel`'s docstring.

## Package layout

```
src/ss_infection/
  __main__.py          command line
  models.py            PDE models (weak forms) and their declarations
  factory.py           model registry built from models.MODEL_CLASSES
  initial_conditions.py
  simulation.py        simulate(), run_simulation(), SimulationResult
  metrics.py           per-step trackers (tip, primed band, signal)
  regimes.py           regime classification and stall detection
  plotting.py          figures and save_run_outputs()
  history.py           history .npz writer
  outputs.py           ProfileGroup / Overlay / SeriesSpec declarations
  sweep.py             parameter sweeps
  dispersion.py        dispersion-relation wave speeds
  extinction.py        analytical extinction estimates
examples/              sweep configurations
```

## Development

```bash
flake8
```

## Known limitations

* No density cutoff: free phage never vanish behind a stalled front (with
  δ_V = 0), so failed invasions are classified `stall` rather than `extinct`.
  Stop-and-go patterns should be checked against a tenfold change in the
  cutoff (manuscript §3.11) before being called physical.
* Runs stop early when the 50% front reaches 80% of the domain.
* The initial phage bolus is fixed at 5·exp(−x²/2).
* `--theoretical` and theoretical sweeps for `delay_chain` apply the default
  `f_abi = 0.5`, which lowers the reported speed; pass `--f_abi 0` for a naive
  lawn.

## License

No license has been chosen yet.
