"""Per-step front trackers.

FrontTracker (every model) records the leading infection edge. Its
subclass SignalFrontTracker (immune signaling models, manuscript
§§ 3.2-3.4) adds the primed band, halo and signal metrics.

Tracker interface (any class a model lists in ``trackers``):
    __init__(model, params, x_coords, S0)
    update(t, field)        every step; field(indices) -> profile array
    record_snapshot(field)  at each cached snapshot
    arrays()                {name: array} written to the history file
    summary()               {column: value}, keys = SUMMARY_COLUMNS
    SUMMARY_COLUMNS         per-run scalars that become CSV columns

Per time step (arrays aligned with the history's ``t``; NaN = undefined):
    tip_x               leading infection edge, farthest x with U < 0.99 S0
                        (every model)
    primed_edge_x       leading primed edge, farthest x with
                        R_m >= S0 / 2
    primed_band_width   total length where R_m >= S0 / 2
    L_halo              primed_edge_x - tip_x, clipped at 0 (0 when no
                        primed band; primed cells only behind the tip
                        give no halo)
    sigma_tip           signal at the tip, F_f (column name kept)
    x_K                 farthest x with F >= K
    z_K                 threshold distance x_K - tip_x (negative when the
                        signal is still below K at the tip)

Per node (arrays aligned with ``x``):
    t_signal_x          first time F(x) >= K
    t_tip_x             first time the tip passes x (U(x) < 0.99 S0)
    warning_x           warning time T(x) = t_tip_x - t_signal_x
                        (negative when the signal arrives after infection)

All edges are linearly interpolated between nodes.
"""

import numpy as np

TIP_LEVEL = 0.99
PRIMED_LEVEL = 0.5

# Per-run summaries, in CSV column order (see factory.output_columns)
SUMMARY_COLUMNS = (
    'L_halo_max', 'L_halo_final',
    'primed_band_width_max', 'primed_band_width_final',
    'sigma_tip_final', 'z_K_final', 'warning_median',
)


def leading_edge(x, f, level):
    """Farthest x where f >= level, interpolated to the crossing.

    Returns NaN when f never reaches the level and x[-1] when it holds
    at the right boundary.
    """
    idx = np.nonzero(f >= level)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[-1]
    if i == len(x) - 1:
        return float(x[-1])
    drop = f[i] - f[i + 1]
    frac = (f[i] - level) / drop if drop > 0 else 0.0
    return float(x[i] + frac * (x[i + 1] - x[i]))


class FrontTracker:
    """Records the leading infection edge (tip) every time step.

    Needs the model to provide get_front_subspaces(). The regime
    classifier (regimes.py) derives speeds and stalls from ``tip_x``.
    """

    SUMMARY_COLUMNS = ()

    SERIES = ('tip_x',)

    def __init__(self, model, params, x_coords, S0):
        self.x = np.asarray(x_coords)
        self.S0 = float(S0)
        self.L = float(params['L'])
        self.front_subs = tuple(model.get_front_subspaces())
        self.series = {name: [] for name in self.SERIES}

    def tip(self, U):
        """Farthest x with U < TIP_LEVEL * S0."""
        return leading_edge(self.x, -U, -TIP_LEVEL * self.S0)

    def update(self, t, field):
        self.series['tip_x'].append(self.tip(field(self.front_subs)))

    def record_snapshot(self, field):
        pass

    def arrays(self):
        """Everything to cache, keyed by history-file name."""
        return dict(self.series)

    def summary(self):
        return {}


class SignalFrontTracker(FrontTracker):
    """Tracks tip, primed band and signal metrics every time step.

    Needs the model to provide get_front_subspaces(),
    get_fully_primed_subspace(), get_signal_subspace() and
    signal_threshold(params).
    """

    SUMMARY_COLUMNS = SUMMARY_COLUMNS

    SERIES = (
        'tip_x', 'primed_edge_x', 'primed_band_width', 'L_halo',
        'sigma_tip', 'x_K', 'z_K',
    )

    def __init__(self, model, params, x_coords, S0):
        super().__init__(model, params, x_coords, S0)
        self.K = float(model.signal_threshold(params))
        self.primed_sub = (model.get_fully_primed_subspace(),)
        self.signal_sub = (model.get_signal_subspace(),)
        self.primed_edge_snap = []
        self.t_signal_x = np.full(len(self.x), np.nan)
        self.t_tip_x = np.full(len(self.x), np.nan)

    def primed_edge(self, R_m):
        return leading_edge(self.x, R_m, PRIMED_LEVEL * self.S0)

    def update(self, t, field):
        self._update(
            t, field(self.front_subs), field(self.primed_sub),
            field(self.signal_sub),
        )

    def _update(self, t, U, R_m, F):
        x = self.x
        tip = self.tip(U)
        primed_mask = R_m >= PRIMED_LEVEL * self.S0
        edge = self.primed_edge(R_m)
        x_K = leading_edge(x, F, self.K)

        if np.isnan(tip):
            halo = np.nan
        elif np.isnan(edge):
            halo = 0.0
        else:
            halo = max(edge - tip, 0.0)

        values = {
            'tip_x': tip,
            'primed_edge_x': edge,
            'primed_band_width': float(
                np.trapezoid(primed_mask.astype(float), x)
            ),
            'L_halo': halo,
            'sigma_tip': (
                float(np.interp(tip, x, F)) if not np.isnan(tip)
                else np.nan
            ),
            'x_K': x_K,
            'z_K': x_K - tip,
        }
        for name, val in values.items():
            self.series[name].append(val)

        new_signal = np.isnan(self.t_signal_x) & (F >= self.K)
        self.t_signal_x[new_signal] = t
        new_tip = np.isnan(self.t_tip_x) & (U < TIP_LEVEL * self.S0)
        self.t_tip_x[new_tip] = t

    def record_snapshot(self, field):
        """Primed edge at a kymograph snapshot (aligned with t_snap)."""
        self.primed_edge_snap.append(
            self.primed_edge(field(self.primed_sub))
        )

    def warning_time(self):
        return self.t_tip_x - self.t_signal_x

    def arrays(self):
        data = super().arrays()
        data['primed_edge_snap'] = self.primed_edge_snap
        data['t_signal_x'] = self.t_signal_x
        data['t_tip_x'] = self.t_tip_x
        data['warning_x'] = self.warning_time()
        return data

    def summary(self):
        """Scalar summaries for the run.

        warning_median skips the first 10% of the domain, where the
        initial bolus rather than the travelling front sets the timing.
        """
        def last(name):
            vals = np.asarray(self.series[name], dtype=float)
            vals = vals[~np.isnan(vals)]
            return float(vals[-1]) if len(vals) else np.nan

        def peak(name):
            vals = np.asarray(self.series[name], dtype=float)
            return float(np.nanmax(vals)) if np.any(~np.isnan(vals)) \
                else np.nan

        warn = self.warning_time()
        keep = (self.x > 0.1 * self.L) & ~np.isnan(warn)
        summary = {
            'L_halo_max': peak('L_halo'),
            'L_halo_final': last('L_halo'),
            'primed_band_width_max': peak('primed_band_width'),
            'primed_band_width_final': last('primed_band_width'),
            'sigma_tip_final': last('sigma_tip'),
            'z_K_final': last('z_K'),
            'warning_median': (
                float(np.median(warn[keep])) if np.any(keep) else np.nan
            ),
        }
        return {key: summary[key] for key in SUMMARY_COLUMNS}
