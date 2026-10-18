import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from . import regimes
from .models import BASE_OUTPUT_COLUMNS
# Figures live in plotting.py; re-exported for existing imports
from .plotting import plot_kymograph, plot_snapshot  # noqa: F401


@dataclass
class SimulationResult:
    """Everything one run produces.

    The same object comes from a live run (``simulate`` /
    ``run_simulation``) or from a cached history file
    (``SimulationResult.from_history``), so plotting and analysis code
    never needs to know which.

    Attributes:
        model     the model instance (supplies plot/overlay declarations)
        params    full parameter dict
        x         sorted node coordinates, shape (nx,)
        t_snap    snapshot times, shape (n_snap,)
        profiles  {group name: array (n_snap, nx)}, always including 'U'
        series    {name: 1D array}: per-step series ('t', 'p_end',
                  'p_total', tracker series), front and fit points,
                  per-node tracker arrays
        metrics   {output column: value} (see model.output_columns())
        success   False if the solver failed (or the run never finished)
    """

    model: object
    params: dict
    x: np.ndarray
    t_snap: np.ndarray
    profiles: dict
    series: dict
    metrics: dict
    success: bool = True

    @property
    def S0(self):
        return float(self.params.get('B0_val', self.params.get('B0', 1.0)))

    def kymograph_field(self):
        """Uninfected density U in % of S0, shape (nx, n_snap)."""
        return 100.0 * np.asarray(self.profiles['U']).T / self.S0

    @classmethod
    def from_history(cls, path):
        """Rebuild a result from a history .npz (finished or .tmp)."""
        from .factory import build_model

        d = np.load(path)
        params = json.loads(str(d['params_json']))
        model, _ = build_model(params)

        if 'profile_names' in d.files:
            names = [str(n) for n in d['profile_names']]
        else:
            names = [k for k in d.files if d[k].ndim == 2]
        profiles = {name: d[name] for name in names}
        # Files written before the manuscript notation (phage 'P', not 'V')
        if 'V' not in profiles and 'P' in profiles:
            legacy = model.legacy_profile_names
            profiles = {legacy.get(k, k): val for k, val in profiles.items()}

        scalar_keys = {'x', 't_snap', 'profile_names', 'params_json'}
        series = {
            k: d[k] for k in d.files
            if k not in scalar_keys and k not in profiles and d[k].ndim == 1
        }

        metrics = {}
        for col in model.output_columns():
            if col == 'p_end':
                metrics[col] = float(d['p_end_max'])
            elif col in d.files:
                val = d[col]
                metrics[col] = str(val) if val.dtype.kind in 'US' \
                    else float(val)
            else:
                metrics[col] = float('nan')

        success = bool(d['complete']) and float(d['success']) == 1.0
        result = cls(
            model=model, params=params, x=d['x'], t_snap=d['t_snap'],
            profiles=profiles, series=series, metrics=metrics,
            success=success,
        )
        # Reclassify with the current thresholds (cheap), so loaded metrics
        # always match what the plots show; also covers files written
        # before regime classification existed.
        analysis = regimes.analyze(result)
        result.series.update(analysis.series())
        result.metrics.update(analysis.metrics)
        return result


def create_mesh_and_space(comm, L, nx, num_components):
    """Creates the 1D interval mesh and the mixed function space."""
    import basix
    from dolfinx import fem, mesh

    domain = mesh.create_interval(comm, nx, [0.0, L])
    P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    elements = [P1] * num_components
    mixed_el = basix.ufl.mixed_element(elements)
    V = fem.functionspace(domain, mixed_el)
    return domain, V


def setup_solver(comm, F, u):
    """Initializes the non-linear Newton solver."""
    from dolfinx.fem.petsc import NonlinearProblem
    from dolfinx.nls.petsc import NewtonSolver

    problem = NonlinearProblem(F, u)
    solver = NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-6
    return solver


def simulate(params, comm=None, history_path=None, verbose=True):
    """Build the model named by ``params['model']`` and run it.

    The single entry point shared by the CLI and sweeps: derives model
    parameters, builds the mesh and solver, applies the initial
    condition, and returns a SimulationResult.
    """
    from dolfinx import fem
    from mpi4py import MPI

    from .factory import get_model_class

    if comm is None:
        comm = MPI.COMM_SELF

    cls = get_model_class(params.get('model', 'delay_chain'))
    cls.prepare_params(params)
    params['nx'] = int(params['L'] / params['dx'])
    model = cls.from_params(params)
    ic = cls.initial_condition()

    domain, V = create_mesh_and_space(
        comm, params['L'], params['nx'], model.num_components
    )
    u = fem.Function(V)
    u_n = fem.Function(V)
    ic.apply(V, u, params)
    u_n.x.array[:] = u.x.array

    F = model.create_weak_formulation(domain, V, u, u_n, params)
    solver = setup_solver(comm, F, u)

    return run_simulation(
        model, V, u, u_n, solver, params,
        verbose=verbose, history_path=history_path,
    )


def run_simulation(
    model, V, u, u_n, solver, params, verbose=True, history_path=None
):
    """Executes time-stepping loop and computes wavespeed and extinction.

    Uses model.get_front_subspaces() to sum bacterial densities for
    wavefront detection. Fronts, phage totals and tracker series are
    recorded every step; the profiles in model.history_groups() every
    ``params['history_every']`` steps. If ``history_path`` is given, they
    are cached there (see ``history.py``).

    Returns:
        SimulationResult
    """
    from .history import HistoryWriter

    # Collapsed dof maps (with coordinate sort) per subspace, built once
    sub_cache = {}

    def sub_map(idx):
        if idx not in sub_cache:
            V_sub, dof_map = V.sub(idx).collapse()
            coords = V_sub.tabulate_dof_coordinates()[:, 0]
            sort_idx = np.argsort(coords)
            sub_cache[idx] = (np.asarray(dof_map)[sort_idx], coords[sort_idx])
        return sub_cache[idx]

    def field(indices):
        return sum(u.x.array[sub_map(i)[0]] for i in indices)

    front_subs = model.get_front_subspaces()
    groups = model.history_groups()

    # Coordinates from the free phage subspace (shared mesh)
    x_coords = sub_map(0)[1]

    S0 = float(params.get('B0_val', params.get('B0', 1.0)))
    p_ext_default = params.get('P_extinction', 1e-6)
    p_ext = float(
        params.get(
            'p_ext', params.get('p_extinction', p_ext_default)
        )
    )
    history_every = max(1, int(params.get('history_every') or 10))

    trackers = [
        tracker(model, params, x_coords, S0) for tracker in model.trackers
    ]

    writer = None
    if history_path is not None:
        writer = HistoryWriter(
            history_path, x_coords, params,
            flush_every=params.get('history_flush') or 20,
        )

    front_positions = []
    times = []
    snap_times = []
    profiles = {name: [] for name in groups}
    p_end_history = []
    p_total_history = []
    time_history = []
    fit_times = np.array([])
    fit_line = np.array([])

    def scalars():
        data = {
            't': time_history,
            'p_end': p_end_history,
            'p_total': p_total_history,
            'front_t': times,
            'front_x': front_positions,
            'fit_t': fit_times,
            'fit_x': fit_line,
        }
        for tracker in trackers:
            data.update(tracker.arrays())
        return data

    def record_step(t):
        P_curr = field([0])
        time_history.append(t)
        p_end_history.append(float(P_curr[-1]))
        p_total_history.append(float(np.trapezoid(P_curr, x_coords)))
        for tracker in trackers:
            tracker.update(t, field)

    def record_snapshot(t):
        snapshot = {name: field(idx) for name, idx in groups.items()}
        snap_times.append(t)
        for name, values in snapshot.items():
            profiles[name].append(values)
        for tracker in trackers:
            tracker.record_snapshot(field)
        if writer is not None:
            writer.record(t, snapshot)
            writer.maybe_flush(scalars())

    # Record initial condition at t = 0.0
    record_step(0.0)
    record_snapshot(0.0)

    t = 0.0
    step = 0
    if verbose:
        print("Starting simulation...")
    mark = datetime.now()
    success = True

    try:
        while t < params['t_end']:
            t += params['dt']
            step += 1
            solver.solve(u)
            u_n.x.array[:] = u.x.array

            # Track distal free phage at x = L_total and total phage
            record_step(t)
            if step % history_every == 0:
                record_snapshot(t)

            # Sum bacterial densities from all tracked subspaces
            B_total = field(front_subs)
            threshold = S0 / 2.0
            invaded = np.where(B_total < threshold)[0]

            if len(invaded) > 0:
                front_x = x_coords[invaded[-1]]
                front_positions.append(front_x)
                times.append(t)

                # Stop early if the front reaches 80% of the domain
                if front_x >= 0.8 * params['L']:
                    if verbose:
                        print(
                            f"Front reached 80% of domain ({front_x:.2f}). "
                            "Stopping early."
                        )
                    break
    except Exception as e:
        if verbose:
            print(f"Solver failed to converge: {e}")
        success = False

    # Always keep the final converged state as the last snapshot
    if success and snap_times[-1] != t:
        record_snapshot(t)

    if not success:
        c_empirical = float('nan')
    else:
        half_idx = len(times) // 2
        if len(times) > 1:
            fit_times = np.array(times[half_idx:])
            fit_positions = np.array(front_positions[half_idx:])
            p = np.polyfit(fit_times, fit_positions, 1)
            c_empirical = float(p[0])
            intercept = float(p[1])
            fit_line = c_empirical * fit_times + intercept
        else:
            c_empirical = 0.0

    # Calculate Extinction Metrics
    p_end_max = max(p_end_history) if p_end_history else 0.0
    if p_end_max >= p_ext:
        extinction = 0
        t_extinction = float('nan')
    else:
        extinction = 1
        t_extinction = float('nan')
        # Find first time step where total phage drops below p_ext after peak
        peak_idx = int(np.argmax(p_total_history))
        for idx in range(peak_idx, len(p_total_history)):
            if p_total_history[idx] < p_ext:
                t_extinction = time_history[idx]
                break
        if np.isnan(t_extinction) and len(time_history) > 1:
            t_extinction = time_history[-1]

    # Run results keyed by output column (see model.output_columns)
    metrics = {
        'c_star': c_empirical,
        'extinction': extinction,
        'p_end': p_end_max,
        't_extinction': t_extinction,
    }
    for tracker in trackers:
        metrics.update(tracker.summary())

    result = SimulationResult(
        model=model,
        params=params,
        x=x_coords,
        t_snap=np.array(snap_times),
        profiles={name: np.vstack(snaps) for name, snaps in profiles.items()},
        series={
            name: np.asarray(values, dtype=float)
            for name, values in scalars().items()
        },
        metrics=metrics,
        success=success,
    )

    # Regime classification (regimes.py) from the finished run
    analysis = regimes.analyze(result)
    result.series.update(analysis.series())
    result.metrics = {
        col: {**metrics, **analysis.metrics}.get(col, float('nan'))
        for col in model.output_columns()
    }

    if verbose:
        print("Simulation complete.")
        if model.reports_wave_speed:
            print(f"Calculated wave speed: c* = {c_empirical:.4f}")
        status_str = "Extinguished" if extinction == 1 else "Breakthrough"
        print(f"Extinction status: {extinction} ({status_str})")
        print(f"Peak distal phage (p_end): {p_end_max:.4e}")
        if extinction == 1:
            print(f"Time of extinction: t = {t_extinction:.2f}")
        for key, val in result.metrics.items():
            if key not in BASE_OUTPUT_COLUMNS:
                print(f"{key}: {_fmt(val)}")
        elapsed = str(datetime.now() - mark)
        print(f"Total runtime: {elapsed}\n")

    if writer is not None:
        summary = dict(result.metrics, p_end_max=p_end_max, success=success)
        del summary['p_end']
        writer.flush(result.series, summary=summary, complete=True)

    return result


def _fmt(val):
    return val if isinstance(val, str) else f"{val:.4g}"
