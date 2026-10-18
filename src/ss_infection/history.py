"""Per-run history caching.

A run's histories are written to ``<path>.tmp`` while it is in progress,
rewritten every ``flush_every`` snapshots, and renamed to ``<path>`` once
the run finishes. Runs that crash or are killed leave a readable partial
``.tmp`` file behind.

File contents (``numpy.load(path)``):
    x                  (nx,)           sorted node coordinates
    t_snap             (n_snap,)       snapshot times (every history_every
                                       steps, plus the final state)
    <group>            (n_snap, nx)    spatial profile of each model group
                                       (e.g. U, V, S, I, J, F, R, L)
    profile_names      (n_groups,)     which keys above are profiles
    t                  (n_steps,)      every time step
    p_end, p_total     (n_steps,)      distal and integrated free phage
    front_t, front_x                   50% front of uninfected cells U
    fit_t, fit_x                       linear fit giving c_star
    signal models only (see ``metrics.py``):
      tip_x, primed_edge_x, primed_band_width, L_halo, sigma_tip, x_K,
      z_K              (n_steps,)  front / halo / signal metrics
      primed_edge_snap (n_snap,)   primed edge at each snapshot
      t_signal_x, t_tip_x, warning_x  (nx,)  per-node warning time T(x)
      L_halo_max, ..., warning_median   summary scalars
    c_star, extinction, p_end_max, t_extinction, success
                                       run summary (NaN while running)
    c_tip, stall       (n_steps,)      smoothed tip speed, stall mask
    regime, c_ref, c_mean, ...         regime summary (regimes.py);
                                       ``regime`` is a string
    complete           bool            False in a .tmp file
    params_json        str             full parameter dict as JSON

Load a finished or partial file with
``simulation.SimulationResult.from_history(path)``.
"""

import json
import os

import numpy as np


class HistoryWriter:
    """Accumulates snapshots and flushes them to a compressed .npz file."""

    def __init__(self, path, x_coords, params, flush_every=20):
        self.path = path
        self.tmp_path = path + '.tmp'
        self.flush_every = max(1, int(flush_every))
        self.x = np.asarray(x_coords)
        self.params_json = json.dumps(params, default=str, sort_keys=True)
        self.t_snap = []
        self.profiles = {}
        self._since_flush = 0

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def record(self, t, profiles):
        """Store one snapshot; ``profiles`` maps group name -> 1D array."""
        self.t_snap.append(float(t))
        for name, values in profiles.items():
            self.profiles.setdefault(name, []).append(np.array(values))
        self._since_flush += 1

    def maybe_flush(self, scalars):
        if self._since_flush >= self.flush_every:
            self.flush(scalars, complete=False)

    def flush(self, scalars, summary=None, complete=False):
        """Write everything recorded so far.

        Writes to a scratch file first and renames it, so a kill during the
        write never leaves a truncated file.
        """
        data = {
            'x': self.x,
            't_snap': np.array(self.t_snap),
            'profile_names': np.array(list(self.profiles), dtype=str),
            'complete': np.array(complete),
            'params_json': np.array(self.params_json),
        }
        for name, snaps in self.profiles.items():
            data[name] = np.vstack(snaps)
        for name, values in scalars.items():
            data[name] = np.asarray(values, dtype=float)
        summary = summary or {}
        for key in ('c_star', 'extinction', 'p_end_max', 't_extinction',
                    'success'):
            data[key] = np.array(float(summary.get(key, np.nan)))
        for key, val in summary.items():
            data[key] = np.array(val if isinstance(val, str) else float(val))

        target = self.path if complete else self.tmp_path
        scratch = target + '.writing'
        with open(scratch, 'wb') as f:
            np.savez_compressed(f, **data)
        os.replace(scratch, target)
        if complete and os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)
        self._since_flush = 0


def history_dir_for(output_csv_path):
    """Directory holding per-run histories for a sweep CSV."""
    return os.path.splitext(output_csv_path)[0] + '_histories'
