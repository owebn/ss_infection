import concurrent.futures
import csv
import glob
import itertools
import json
import math
import multiprocessing
import os

from .dispersion import calculate_theoretical_speed
from .factory import (
    PARAM_ALIASES,
    get_model_class,
    infer_model,
    output_columns,
)
from .history import history_dir_for

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def adaptive_params(params, verbose=True):
    """Applies the model's adaptive overrides for L, t_end, dx, and dt.

    See ``BaseModel.adaptive_overrides``.
    """
    cls = get_model_class(params.get('model', 'delay_chain'))
    old = {key: params.get(key) for key in ('L', 'dx', 't_end', 'dt')}
    params.update(cls.adaptive_overrides(params))
    params['nx'] = int(params['L'] / params['dx'])

    if verbose:
        print("Adapted simulation conditions:")
        for key, fmt in (('L', '.2f'), ('dx', '.4f'), ('t_end', '.2f'),
                         ('dt', '.4f')):
            before = 'unset' if old[key] is None else format(old[key], fmt)
            print(f"  {key}: {before} -> {params[key]:{fmt}}")
        print(f"  nx: {params['nx']}")


def math_linspace(start, end, num):
    """Generates linearly spaced floats without requiring numpy."""
    if num == 1:
        return [start]
    step = (end - start) / (num - 1)
    return [start + i * step for i in range(num)]


def generate_parameter_range(spec):
    """Generates a list of parameter values from a JSON range specification.

    Supports linear and logarithmic scaling.

    Options in spec:
        - values (list): Explicit values, used as given (overrides the
          range options below), e.g. {"values": [4, 6, 8, 12, 16, 24]}.
        - start (float): Starting value.
        - end (float): Ending value.
        - scale (str): 'linear' (default) or 'log' / 'log10'.
        - log (bool): Alternative flag to set scale='log'.
        - step (float): Step size (linear units for linear scale,
          log10 units for log scale).
        - num / num_points (int): Number of points to generate.
    """
    if 'values' in spec:
        return [float(v) for v in spec['values']]

    start = float(spec['start'])
    end = float(spec['end'])
    scale = str(spec.get('scale', 'linear')).lower()
    is_log = (scale in ['log', 'log10']) or bool(spec.get('log', False))

    num = spec.get('num', spec.get('num_points', None))

    if is_log:
        if start <= 0 or end <= 0:
            raise ValueError(
                f"Log scale start ({start}) and end ({end}) must be > 0."
            )
        log_start = math.log10(start)
        log_end = math.log10(end)

        if num is not None:
            log_vals = [
                float(v) for v in math_linspace(log_start, log_end, int(num))
            ]
        else:
            step = float(spec.get('step', 1.0))
            log_vals = []
            curr = log_start
            while curr <= log_end + 1e-9:
                log_vals.append(curr)
                curr += step
        vals = [float(10**lv) for lv in log_vals]
    else:
        if num is not None:
            vals = [
                float(v) for v in math_linspace(start, end, int(num))
            ]
        else:
            step = float(spec.get('step', 1.0))
            vals = []
            curr = start
            while curr <= end + 1e-9:
                vals.append(curr)
                curr += step

    return vals


def _resolve_run_params(combo, keys, constants):
    """Build one run's parameter dict: aliases, model inference, defaults.

    Aliases, defaults and the model implied by parameter names all come
    from the model classes (see ``models.BaseModel``).
    """
    run_params = constants.copy()
    for k, val in zip(keys, combo):
        run_params[k] = val

    for old_k, new_k in PARAM_ALIASES.items():
        if old_k in run_params and new_k not in run_params:
            run_params[new_k] = run_params[old_k]

    if 'model' not in constants:
        inferred = infer_model(list(keys) + list(run_params.keys()))
        if inferred is not None:
            run_params['model'] = inferred

    cls = get_model_class(run_params.get('model', 'delay_chain'))
    for dkey, dval in cls.sweep_defaults.items():
        if dkey not in run_params:
            run_params[dkey] = dval

    cls.prepare_params(run_params)
    return run_params


def _run_single_combination(task_data):
    """Top-level worker function for parallel process execution.

    Returns (combo, metrics) where metrics maps output column -> value.
    """
    (
        combo, keys, constants, adaptive, is_theoretical, history_path
    ) = task_data

    run_params = _resolve_run_params(combo, keys, constants)

    if is_theoretical:
        from .extinction import calculate_extinction_metrics
        c_star, _ = calculate_theoretical_speed(run_params)
        ext, p_end, t_ext = calculate_extinction_metrics(run_params)
        return combo, {
            'c_star': c_star, 'extinction': ext, 'p_end': p_end,
            't_extinction': t_ext,
        }

    if adaptive:
        adaptive_params(run_params, verbose=False)
    else:
        run_params['nx'] = int(run_params['L'] / run_params['dx'])

    from mpi4py import MPI as worker_MPI

    from .simulation import simulate

    result = simulate(
        run_params, comm=worker_MPI.COMM_SELF,
        history_path=history_path, verbose=False,
    )
    return combo, result.metrics


def _prepare_history_dir(history_dir, force):
    """Create the per-run history directory, clearing old runs on force."""
    old_runs = (
        glob.glob(os.path.join(history_dir, 'run_*.npz*'))
        if os.path.isdir(history_dir) else []
    )
    if old_runs and not force:
        raise FileExistsError(
            f"History directory '{history_dir}' already contains runs. "
            "Use --force to overwrite."
        )
    for path in old_runs:
        os.remove(path)
    os.makedirs(history_dir, exist_ok=True)


def run_sweep(
    json_path, output_csv_path, n_jobs=None, theoretical=False, force=False,
    save_histories=True,
):
    """Executes a parameter sweep defined by a JSON configuration file.

    For PDE sweeps, each run's histories are cached in
    ``<output stem>_histories/run_<index>.npz`` (see ``history.py``) and
    the CSV gains a ``history_file`` column pointing to it. Set
    ``"save_histories": false`` in the JSON (or pass --no_histories) to
    disable, and ``history_every`` / ``history_flush`` in ``constants``
    to control the snapshot and flush intervals.
    """
    if os.path.exists(output_csv_path) and not force:
        raise FileExistsError(
            f"Output file '{output_csv_path}' already exists. "
            "Use --force to overwrite."
        )

    if n_jobs is None or n_jobs < 1:
        n_jobs = os.cpu_count() or 1

    rank = 0
    comm = None
    if MPI is not None:
        try:
            comm = MPI.COMM_WORLD
            rank = comm.rank
        except Exception:
            comm = None

    if rank == 0:
        with open(json_path, 'r') as f:
            config = json.load(f)
    else:
        config = None

    if comm is not None:
        config = comm.bcast(config, root=0)

    sweep_spec = config.get('sweep', {})
    constants = config.get('constants', {})
    adaptive = config.get('adaptive', True)
    is_theoretical = (
        theoretical or config.get('theoretical', False)
    )

    keys = list(sweep_spec.keys())
    ranges = [generate_parameter_range(sweep_spec[k]) for k in keys]

    combinations = list(itertools.product(*ranges))
    total_runs = len(combinations)

    mode_desc = (
        "theoretical dispersion calculation"
        if is_theoretical else "PDE simulation"
    )

    # Columns: union over the models in this sweep, in first-seen order.
    # Rows of a model without a column get NaN there.
    out_cols = []
    for combo in combinations:
        model_name = _resolve_run_params(combo, keys, constants).get(
            'model', 'delay_chain'
        )
        for col in output_columns(model_name, is_theoretical):
            if col not in out_cols:
                out_cols.append(col)
    metric_cols = list(out_cols)

    save_histories = (
        save_histories and config.get('save_histories', True)
        and not is_theoretical
    )
    history_paths = [None] * total_runs
    if save_histories:
        history_dir = history_dir_for(output_csv_path)
        if rank == 0:
            _prepare_history_dir(history_dir, force)
        width = max(5, len(str(total_runs - 1)))
        history_paths = [
            os.path.join(history_dir, f"run_{i:0{width}d}.npz")
            for i in range(total_runs)
        ]
        out_cols = out_cols + ['history_file']
    csv_dir = os.path.dirname(os.path.abspath(output_csv_path))

    def csv_row(combo, metrics, history_path):
        row = list(combo) + [
            metrics.get(col, float('nan')) for col in metric_cols
        ]
        if save_histories:
            row.append(os.path.relpath(history_path, csv_dir))
        return row

    if rank == 0:
        print(
            f"Starting parameter sweep ({mode_desc}) with {total_runs} runs "
            f"on {n_jobs} parallel worker(s)..."
        )
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(keys + out_cols)

    tasks = [
        (combo, keys, constants, adaptive, is_theoretical, hist_path)
        for combo, hist_path in zip(combinations, history_paths)
    ]

    if n_jobs == 1:
        for idx, task in enumerate(tasks):
            combo, metrics = _run_single_combination(task)
            c_star, ext = metrics['c_star'], metrics['extinction']
            if rank == 0:
                p_str = ", ".join(f"{k}={v}" for k, v in zip(keys, combo))
                print(
                    f"Completed {idx + 1}/{total_runs}: "
                    f"{p_str} -> c* = {c_star:.4f}, ext = {ext}"
                )
                with open(output_csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(csv_row(combo, metrics, task[5]))
    else:
        mp_context = multiprocessing.get_context('spawn')
        completed_count = 0

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=mp_context
        ) as executor:
            future_to_task = {
                executor.submit(_run_single_combination, task): task
                for task in tasks
            }

            for future in concurrent.futures.as_completed(future_to_task):
                combo, metrics = future.result()
                c_star, ext = metrics['c_star'], metrics['extinction']
                completed_count += 1
                if rank == 0:
                    p_str = ", ".join(
                        f"{k}={v}" for k, v in zip(keys, combo)
                    )
                    print(
                        f"Completed {completed_count}/{total_runs}: "
                        f"{p_str} -> c* = {c_star:.4f}, ext = {ext}"
                    )
                    with open(output_csv_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(
                            csv_row(
                                combo, metrics, future_to_task[future][5]
                            )
                        )

    if rank == 0:
        print(
            "\nParameter sweep finished. "
            f"Results written to {output_csv_path}"
        )
