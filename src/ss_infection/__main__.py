import argparse
import os

from .dispersion import calculate_theoretical_speed
from .factory import MODEL_NAMES, get_model_class
from .sweep import run_sweep


def parse_arguments():
    """Parses command line arguments for the simulation."""
    desc = (
        "1D Spatially Structured Bacteriophage"
        " Infection Simulation"
    )
    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Theoretical mode flag
    parser.add_argument(
        "--theoretical",
        action="store_true",
        help="Calculate theoretical wavespeed from dispersion relation "
             "instead of running PDE simulation",
    )

    # Sweep configuration
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Path to JSON file specifying parameter sweep bounds",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="sweep_results.csv",
        help="Path to output CSV file for sweep results",
    )
    parser.add_argument(
        "--n_jobs",
        "-j",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel workers for sweeps",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Display spatial snapshot profiles instead of kymograph",
    )

    # History caching
    parser.add_argument(
        "--history_every", type=int, default=10,
        help="Record spatial profiles every N time steps",
    )
    parser.add_argument(
        "--history_flush", type=int, default=20,
        help="Rewrite the in-progress history file every N snapshots",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Single runs: write history.npz, kymograph/speed/phage/"
             "snapshot figures, metrics.json and params.json to this "
             "folder (sweeps cache histories to <output stem>_histories/)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Also display the figures when writing --output_dir",
    )
    parser.add_argument(
        "--from_history", type=str, default=None,
        help="Re-plot a cached run's history .npz instead of simulating "
             "(into --output_dir, or shown on screen)",
    )

    # Regime classification thresholds (see regimes.py; default: auto)
    regime = parser.add_argument_group("regime classification")
    regime.add_argument(
        "--regime_transient", type=float, default=None,
        help="Time skipped after the bolus "
             "(default max(3 tau, 10%% of run), at most 50%%)",
    )
    regime.add_argument(
        "--regime_window", type=float, default=None,
        help="Speed-fit window (default max(tau, 20 dt))",
    )
    regime.add_argument(
        "--stall_frac", type=float, default=None,
        help="Stall when tip speed < stall_frac * c_ref (default 0.25)",
    )
    regime.add_argument(
        "--stall_min", type=float, default=None,
        help="Minimum stall duration (default: the speed-fit window)",
    )
    regime.add_argument(
        "--cv_steady", type=float, default=None,
        help="Max speed coefficient of variation for a wave (default 0.15)",
    )
    regime.add_argument(
        "--halo_tol", type=float, default=None,
        help="L_halo above which a wave carries a halo (default 1.0)",
    )
    parser.add_argument(
        "--no_histories",
        action="store_true",
        help="Do not cache per-run histories during sweeps",
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_NAMES),
        default="delay_chain",
        help="PDE model to simulate",
    )

    # Physical parameters
    parser.add_argument(
        "--tau", type=float, default=5.0,
        help="Mean delay period",
    )
    parser.add_argument(
        "--n_stages", type=int, default=10,
        help="Number of latent stages",
    )
    parser.add_argument(
        "--D_val", type=float, default=1.0,
        help="Phage diffusion coefficient D",
    )
    parser.add_argument(
        "--f_B_val", "--beta", type=float, default=1.0,
        help="Adsorption rate beta onto uninfected cells",
    )
    parser.add_argument(
        "--b_val", type=float, default=5.0,
        help="Burst size b",
    )
    parser.add_argument(
        "--B0_val", "--S0", type=float, default=1.0,
        help="Initial lawn density S0",
    )

    # Abi-specific parameters
    parser.add_argument(
        "--k_abi", type=int, default=None,
        help="Abi death stage (defaults to n_stages if None)",
    )
    parser.add_argument(
        "--f_abi", type=float, default=0.5,
        help="Fraction of lawn carrying Abi",
    )

    # Spatial Heterogeneity parameters
    parser.add_argument(
        "--L_naive", "--l_naive", type=float, default=50.0,
        help="Length of naive non-abortive region",
    )
    parser.add_argument(
        "--L_primed", "--l_primed", type=float, default=50.0,
        help="Length of primed abortive patch",
    )
    parser.add_argument(
        "--p_ext", "--P_extinction", "--V_ext", type=float, default=1e-6,
        help="Phage extinction/breakthrough density threshold V_ext",
    )
    parser.add_argument(
        "--cutoff", action="store_true",
        help="Density cutoff (manuscript Methods): phage below V_ext "
             "(--p_ext) neither adsorb nor infect and are removed at "
             "--cutoff_rate; suppresses non-physical tail tunnelling",
    )
    parser.add_argument(
        "--cutoff_rate", type=float, default=5.0,
        help="Removal rate of phage below V_ext when --cutoff is set",
    )

    # Signalling models: death signal (first review, revision of
    # Oct 24 2026, § 3.1). Defaults for the host response are set A'.
    parser.add_argument(
        "--D_signal", "--D_sig", "--D_s", type=float, default=1.0,
        help="Signal diffusion coefficient D_s",
    )
    parser.add_argument(
        "--gamma_signal", "--gamma_sig", "--gamma_s", "--delta_F",
        type=float, default=0.1,
        help="Signal decay rate delta_F",
    )
    parser.add_argument(
        "--T_signal", "--T_sig", "--K", type=float, default=0.1,
        help="Signal threshold K of the RNA response H(F)",
    )
    parser.add_argument(
        "--signal_yield", "--yield_signal", "--M", type=float, default=1.0,
        help="Signal M released per lysing cell",
    )
    parser.add_argument(
        "--hill_signal", type=float, default=20.0,
        help="Hill steepness h of H(F)",
    )

    # RNA-protein host response (eq. 22): dm/dt = alpha_m H(F) - delta_m m,
    # dp/dt = alpha_p m - delta_p p, resistant share phi(p)
    parser.add_argument(
        "--alpha_m", type=float, default=1.0,
        help="Maximal RNA production rate alpha_m",
    )
    parser.add_argument(
        "--delta_m", type=float, default=1.0,
        help="RNA decay rate delta_m",
    )
    parser.add_argument(
        "--alpha_p", type=float, default=2.0,
        help="Protein production per RNA alpha_p",
    )
    parser.add_argument(
        "--delta_p", type=float, default=0.1,
        help="Protein decay rate delta_p (0 = permanent priming)",
    )
    parser.add_argument(
        "--phi_max", type=float, default=1.0,
        help="Maximum resistant proportion phi_max",
    )
    parser.add_argument(
        "--K_p", type=float, default=1.0,
        help="Half-saturation protein level K_p of phi(p)",
    )
    parser.add_argument(
        "--response_timescale", "--tau_R", type=float, default=1.0,
        help="Divides alpha_m, delta_m, alpha_p and delta_p: stretches the "
             "priming curve without changing p_inf or phi_inf",
    )
    parser.add_argument(
        "--n_R", type=int, default=None,
        help="Last latent stage whose cells can still switch "
             "(0 = status fixed at infection; default n_stages)",
    )
    parser.add_argument(
        "--b_abi", "--b_S", "--b_R", type=float, default=0.0,
        help="Burst b_R of resistant (abortive) cells",
    )
    parser.add_argument(
        "--delta_P", "--delta_V", type=float, default=0.0,
        help="Free phage decay rate delta_V",
    )
    parser.add_argument(
        "--f_I_val", "--beta_I", type=float, default=None,
        help="Adsorption beta_I to infected cells (defaults to beta)",
    )

    # Phage signal sensing (both evasion models; review § 4.5 / § 6.4)
    parser.add_argument(
        "--sigma_p", "--F_p", type=float, default=None,
        help="Phage signal trigger F_p (defaults to K)",
    )
    parser.add_argument(
        "--hill_p", "--h_p", type=float, default=None,
        help="Hill steepness h_p of the phage trigger (defaults to h)",
    )

    # Latency / burst evasion (immune_signaling_evasion; review § 4)
    parser.add_argument(
        "--tau_slow", type=float, default=None,
        help="Latent period at high signal (defaults to tau)",
    )
    parser.add_argument(
        "--b_slow", type=float, default=None,
        help="Burst at high signal, burst_mode=switch (defaults to b_val)",
    )
    parser.add_argument(
        "--burst_mode", type=str, choices=["switch", "frontier"],
        default="switch",
        help="'switch': Hill switch b -> b_slow; 'frontier': burst follows "
             "latency via b_val * (1 - exp(-(tau - tau_e) / tau_m))",
    )
    parser.add_argument(
        "--tau_eclipse", type=float, default=0.2,
        help="Eclipse period tau_e (burst_mode=frontier)",
    )
    parser.add_argument(
        "--tau_assembly", type=float, default=1.0,
        help="Assembly time tau_m (burst_mode=frontier)",
    )

    # Lysogeny (lysogeny_delay_chain, immune_signaling_lysogeny_evasion)
    parser.add_argument(
        "--p_lys", "--ell", type=float, default=0.0,
        help="Lysogeny probability ell (at low signal for the evasion "
             "model)",
    )
    parser.add_argument(
        "--p_lys_hi", "--ell_hi", type=float, default=None,
        help="Lysogeny probability ell_hi at high signal (defaults to "
             "p_lys)",
    )
    parser.add_argument(
        "--nu_induction", "--nu", "--gamma", type=float, default=0.02,
        help="Prophage induction rate gamma",
    )
    parser.add_argument(
        "--f_L_val", "--beta_L", type=float, default=None,
        help="Adsorption beta_L to lysogens (defaults to beta)",
    )
    parser.add_argument(
        "--lys_window_start", type=int, default=1,
        help="First latent stage where lysogeny is allowed (1-based)",
    )
    parser.add_argument(
        "--lys_window_end", type=int, default=None,
        help="Last latent stage where lysogeny is allowed (default n_stages)",
    )

    # Simulation parameters
    parser.add_argument(
        "--dt", type=float, default=0.1,
        help="Time step size",
    )
    parser.add_argument(
        "--t_end", type=float, default=25.0,
        help="Total simulation time",
    )
    parser.add_argument(
        "--dx", type=float, default=0.1,
        help="Space step size",
    )
    parser.add_argument(
        "--L", type=float, default=200.0,
        help="Length of the 1D domain",
    )

    return parser.parse_args()


def main():
    """Main execution function for the simulation."""
    args = parse_arguments()

    if args.sweep is not None:
        run_sweep(
            args.sweep,
            args.output,
            n_jobs=args.n_jobs,
            theoretical=args.theoretical,
            force=args.force,
            save_histories=not args.no_histories,
        )
        return

    if args.from_history is not None:
        from .simulation import SimulationResult
        result = SimulationResult.from_history(args.from_history)
        _apply_regime_overrides(result, args)
        _outputs(result, args)
        return

    params = vars(args)
    model_cls = get_model_class(params['model'])

    if args.theoretical:
        c_star, lambda_star = calculate_theoretical_speed(params)
        print("\n--- Theoretical Wave Speed Calculation ---")
        print(f"Model: {params['model']}")
        print(
            f"Parameters: n={params['n_stages']}, tau={params['tau']}, "
            f"D={params['D_val']}, f_B={params['f_B_val']}, "
            f"B0={params['B0_val']}, b={params['b_val']}"
        )
        for line in model_cls.describe_theory(params):
            print(line)
        print(f"Calculated theoretical wave speed: c* = {c_star:.4f}")
        print(f"Spatial decay rate: lambda* = {lambda_star:.4f}\n")
        return

    # Defer dolfinx imports so --theoretical mode works without dolfinx
    from mpi4py import MPI

    from .simulation import simulate

    comm = MPI.COMM_WORLD
    history_path = None
    if args.output_dir is not None and comm.rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        history_path = os.path.join(args.output_dir, 'history.npz')
    result = simulate(params, comm=comm, history_path=history_path)

    if comm.rank == 0:
        _outputs(result, args)


REGIME_FLAGS = (
    'regime_transient', 'regime_window', 'stall_frac', 'stall_min',
    'cv_steady', 'halo_tol',
)


def _apply_regime_overrides(result, args):
    """Re-classify a loaded run with any regime thresholds given here."""
    from . import regimes

    overrides = {
        key: getattr(args, key) for key in REGIME_FLAGS
        if getattr(args, key) is not None
    }
    if not overrides:
        return
    result.params.update(overrides)
    analysis = regimes.analyze(result)
    result.series.update(analysis.series())
    result.metrics.update(analysis.metrics)


def _outputs(result, args):
    """Write every figure to --output_dir, or show one on screen."""
    from .plotting import plot_kymograph, plot_snapshot, save_run_outputs

    if args.output_dir is not None:
        written = save_run_outputs(result, args.output_dir, show=args.show)
        print(f"Wrote {len(written)} files to {args.output_dir} "
              f"(regime: {result.metrics.get('regime')})")
    elif args.snapshot:
        plot_snapshot(result)
    else:
        plot_kymograph(result)


if __name__ == "__main__":
    main()
