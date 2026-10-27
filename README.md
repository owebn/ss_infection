# ss_infection

One-dimensional reaction–diffusion simulations (FEniCSx) and dispersion-relation
theory for phage infection fronts spreading through a bacterial lawn. The
models cover an Erlang-distributed latent period, abortive-infection (Abi)
cells, a diffusible interferon-like death signal that primes neighbouring
hosts, phage that read that signal to change latency, burst or lysogeny, and
temperate phage with a prophage bank.

Notation follows the *First Review* (Wingreen group), revision of Oct 24 2026:
V free phage, U uninfected cells (S naive, R resistant), F the signal, β
adsorption. The host response is an RNA–protein field: in each spatial bin the
signal drives RNA m, RNA drives protein p, and the protein sets the resistant
share φ(p) of every cell that can still switch (§ 3.1).

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

The review's stop-and-go parameter set B′ (§ 8.1):

```bash
python -m ss_infection --model immune_signaling --n_stages 1 --tau 0.5 --M 40 --K 1 --alpha_m 4 --delta_m 2 --alpha_p 4 --delta_p 0.4 --D_s 40 --delta_F 0.5 --b_R 0.5 --n_R 1 --cutoff --p_ext 1e-4 --L 300 --t_end 200 --dt 0.02 --history_every 50 --output_dir runs/setB
```

Priming-curve and ceiling numbers for a parameter set (t₁/₂, t⋆, t_rev, φ∞,
b∞_eff) are printed by `--theoretical`.

`python -m ss_infection --help` lists every option. After installation the
same command is also available as `ss-infection`.

## Models

| `--model` | Description | Manuscript |
|---|---|---|
| `delay_chain` | Phage V, susceptible cells S, latent chain I₁…Iₙ (rate w = n/τ, burst b); infected cells shield at β_I | §1.1 |
| `delay_fixed` | As `delay_chain` with an exact latent period τ (the n → ∞ limit at fixed mean): each infection bursts τ later, read from a buffer of past infection fluxes (τ rounded to whole steps) | §1.4 |
| `abi` | Lawn with a fraction `f_abi` of abortive cells R (burst b_R, default 0); optional phage decay δ_V; `barrier_depleting = 0` makes R a non-depleting sink (not consumed on adsorption) | §2.1 |
| `abi_heterogeneous` | Naive region of length `L_naive` followed by an abortive patch of length `L_primed`, then `L_after` of naive lawn (default 0). With `f_abi = 1` and `L_after > 0` the patch is a barrier of length L | §2.4 |
| `lysogeny_delay_chain` | Temperate phage: constant lysogeny probability `p_lys` over a decision window; lysogens L induce at rate ν | §6.1 (no signal) |
| `immune_signaling` | Death signal F (yield M, diffusion D_s, decay δ_F) drives a non-diffusing RNA–protein response, ṁ = α_m H(F) − δ_m m, ṗ = α_p m − δ_p p; cells switch fast and reversibly to resistance with share φ(p) = φ_max p/(K_p + p). Uninfected cells and latent stages k ≤ n_R switch; later stages freeze. Reversion is intrinsic (protein decay) | §§3.1–3.7 |
| `immune_signaling_evasion` | As above; phage switch latency (`--tau_slow`) and burst (`--b_slow` or `--burst_mode frontier`) when F exceeds the trigger F_p | §4.5 |
| `immune_signaling_lysogeny_evasion` | As above; phage switch lysogeny probability from ℓ (`p_lys`) to ℓ_hi (`p_lys_hi`) when F exceeds F_p. Lysogens don't switch; an induced lysogen takes the local φ(p) | §6 |
| `immune_signaling_switching` | `immune_signaling` with a finite switching rate κ (`kappa_switch`): explicit naive S and resistant R, S → R at κφ(p), R → S at κ(1 − φ(p)); latent stages k ≤ n_R switch I_k ↔ J_k at the same rates. κ → ∞ recovers `immune_signaling` | §9, open problem 7 |

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
| δ_F, K, M, h | `gamma_signal`, `T_signal`, `signal_yield`, `hill_signal` | `delta_F`, `K`, `M`, `h`* |
| α_m, δ_m, α_p, δ_p | `alpha_m`, `delta_m`, `alpha_p`, `delta_p` | – |
| φ_max, K_p, n_R | `phi_max`, `K_p`, `n_R` (default n) | – |
| host time scale | `response_timescale` (divides α_m, δ_m, α_p, δ_p) | `tau_R` |
| ℓ, ℓ_hi, γ | `p_lys`, `p_lys_hi`, `nu_induction` | `ell`, `ell_hi`, `gamma` |
| F_p, h_p | `sigma_p`, `hill_p` | `F_p`, `h_p` |
| δ_V, V_ext | `delta_P`, `p_ext` | `delta_V`, `V_ext` |
| barrier L, run-up | `L_primed`, `L_naive` | `L_barrier`, `run_up` |
| κ | `kappa_switch` | `kappa` |

\* sweep JSON and Python only (no one-letter command-line flag). `alpha` is
reserved as an older alias of `f_abi`. Host-response defaults are set A′
(α_m = 1, δ_m = 1, α_p = 2, δ_p = 0.1, φ_max = K_p = 1).

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
`"scale": "log"` optional; or an explicit list `{"values": [4, 6, 8]}`),
`constants`, and `adaptive` (domain size and
resolution chosen from a Fisher–KPP estimate). See `examples/`. Each run's
history goes to `results_histories/run_XXXXX.npz`, linked from the CSV's
`history_file` column (disable with `--no_histories`).

CSV columns depend on the model:

* every run: `c_star`, `extinction`, `p_end`, `t_extinction`
* PDE runs: regime metrics `regime`, `c_ref`, `c_mean`, `c_cv`, `X_max`,
  `t_arrest`, `n_stalls`, `T_cycle`, `L_step`, `hit_domain_end`
* signal models: `L_halo_max/final` (tip to the R = φU ≥ S₀/2 edge),
  `L_halo_mid_max/final` (the review's halo: U = ½ front to φ = ½),
  `primed_band_width_max/final`,
  `sigma_tip_final` (edge signal F_f), `z_K_final`, `warning_median`
* `abi_heterogeneous`: barrier metrics `V_in_max` (entrance density),
  `N_reach_final` (cumulative net phage flux into the barrier),
  `X_eat_final` (depth with R < f_abi S₀/2), `t_breach` (first time the tip
  passes the far side; needs `L_after > 0`), with per-step series `V_in`,
  `N_reach`, `X_eat` in the history file

### Regime sweeps (`examples/regimes/`)

Sweeps built on the review's regime map for a non-adaptive front with the
RNA–protein host response (parameter sets A′ and B′, §§ 3.6–3.13, § 8.1).
Each file carries a `description` and the `expected` regimes, which the sweep
loader ignores.

| File | Sweep | Expected regimes |
|---|---|---|
| `setA_stall_vs_wave.json` | D_s 2–14, b_R = 0.5 (7 runs) | `wave_fast` below D_s ≈ 4.2, `stall` (arrest then creep) above |
| `setA_slow_wave_vs_wave.json` | D_s 2–14, b_R = 2 (7 runs) | `wave_fast` below D_s ≈ 4.9, `wave_halo` (ceiling slow wave, c ≈ 1.26) above |
| `setA_regime_map.json` | D_s 2–14 × b_R 0.25–3 (42 runs) | fast / stall (b_R < 0.80, where b∞_eff < 1) / ceiling slow wave |
| `setB_protein_turnover.json` | protein decay δ_p 0.004–4, log (7 runs) | `stall` for slow turnover, `stop_and_go` for δ_p ≈ 0.04–0.4, steady ceiling slow wave for δ_p ≳ 1.3 (φ∞ < f_c) |
| `latency_maturation_map.json` | latency τ 0.1–4 × host time scale 0.1–30, n = 4 (64 runs) | theory-based: `wave_fast` for slow response, `stop_and_go` in between; long latency tests conversion-driven stall vs a silent front |
| `latency_maturation_map_n1.json` | as above with n = 1 (64 runs) | as above, exponential latency |

```bash
python -m ss_infection --sweep examples/regimes/setB_protein_turnover.json -o runs/setB_turnover.csv -j 7
```

These sweeps enable the density cutoff (`"cutoff": true`, see below) and use
`dt = 0.02`, as the manuscript's thresholds were computed with a cutoff and
small time steps. Each run takes roughly 10–20 minutes on one core.

### Front regimes

`regime` is one of `extinct`, `silent`, `stop_and_go`, `stall`,
`single_stall`, `wave`, `wave_fast`, `wave_halo`, `unsteady`, `undetermined`
or `failed`. It is computed from a sliding-window fit of the leading
infection edge: a stall is a stretch where the tip speed stays below
`stall_frac × c_ref` (default 0.25) for at least `stall_min`. All thresholds
can be set with `--regime_transient`, `--regime_window`, `--stall_frac`,
`--stall_min`, `--cv_steady` and `--halo_tol`. See `regimes.py`.

### Starting conditions and switches

* `bolus_amplitude` (default 5): peak of the initial phage bolus A·exp(−x²/2)
  (all models).
* `rescale_time`, `rescale_factor` (all models): multiply free phage V by A
  once at t₀. `rescale_infected = 1` also scales the infected cells (the
  whole linear tip state), which gives the clean lag ln(1/A)/λ*; V alone
  gives a smaller lag, because infections already committed at the tip
  still burst at full size.

Signal models only:

* `signal_on_time`: no signal is released before this time, e.g. to let a
  fast front establish in a naive lawn before signalling starts.
* `primed_patch_p`, `primed_patch_start`, `primed_patch_end`: a pre-primed
  patch with protein p = p₀ on [start, end] at t = 0.

History files of signal models and `delay_chain` also cache `lysing` (the
last latent stage; the lysis flux is w × lysing); signal models also cache
the derived profiles `S`, `R`, `phi`.

### Density cutoff

`--cutoff` (or `"cutoff": true` in a sweep) represents phage discreteness as
in the manuscript's Methods: below V_ext (`--p_ext`) phage neither adsorb nor
infect, and are removed at `--cutoff_rate` (default 5). Without it, a deterministic
front can never truly stop: sub-particle phage tails tunnel through primed
patches and re-ignite fronts beyond them. The cutoff also slows fronts by the
Brunet–Derrida shift (n = 1, w = 2, V_ext = 10⁻⁴: c ≈ 1.98 against c* = 2.12).
It is off by default, which leaves results unchanged.

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
examples/              sweep configurations (regimes/: manuscript regime maps)
```

## Development

```bash
flake8
```

## Known limitations

* Without `--cutoff`, free phage never vanish behind a stalled front (with
  δ_V = 0) and stalled fronts can restart by tail tunnelling. With it,
  stop-and-go patterns should still be checked against a tenfold change in
  V_ext (manuscript §3.11) before being called physical.
* Time stepping is first-order implicit Euler: `dt = 0.05` overestimates front
  speeds by about 7% (n = 1, w = 2); use `dt ≤ 0.02` for quantitative work
  (+3% at 0.02, +0.7% at 0.005). `dispersion.solve_dispersion_discrete(n, D,
  k, w, b, dt, dx)` gives the scheme's own selected speed for comparison.
* Runs stop early when the 50% front reaches 80% of the domain.
* `--theoretical` and theoretical sweeps for `delay_chain` apply the default
  `f_abi = 0.5`, which lowers the reported speed; pass `--f_abi 0` for a naive
  lawn.

## License

No license has been chosen yet.
