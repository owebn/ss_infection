"""Per-step front trackers.

FrontTracker (every model) records the leading infection edge. Its
subclass SignalFrontTracker (immune signaling models, manuscript
§§ 3.2-3.4) adds the primed band, halo and signal metrics, and
BarrierTracker (abi_heterogeneous, review § 2.4) the barrier metrics.

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
    primed_edge_x       leading primed edge, farthest x where the resistant
                        uninfected density R = phi(p) U >= S0 / 2
    primed_band_width   total length where R >= S0 / 2
    L_halo_mid          review's halo (§ 3.5): from the U = S0/2 front to
                        the leading point where phi(p) = 1/2, clipped at 0
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

Barrier (BarrierTracker; barrier = patch on [x_b, x_e], x_b = L_naive,
x_e = x_b + L_primed):
    V_in                phage density at the barrier entrance V(x_b)
    N_reach             cumulative net phage flux into the barrier,
                        integral of -D dV/dx at x_b over time
    X_eat               depth eaten, from x_b to the farthest barrier point
                        where the abortive cells R < f_abi S0 / 2 (0 for a
                        non-depleting sink)
    t_breach (summary)  first time the tip passes x_e (needs L_after > 0;
                        NaN if never)

All edges are linearly interpolated between nodes.
"""

import numpy as np

TIP_LEVEL = 0.99
PRIMED_LEVEL = 0.5

# Per-run summaries, in CSV column order (see factory.output_columns)
SUMMARY_COLUMNS = (
    'L_halo_max', 'L_halo_final', 'L_halo_mid_max', 'L_halo_mid_final',
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


class BarrierTracker(FrontTracker):
    """Tip plus barrier entrance density, phage budget and eaten depth.

    Needs the model to provide get_abortive_subspace().
    """

    SUMMARY_COLUMNS = (
        'V_in_max', 'N_reach_final', 'X_eat_final', 't_breach',
    )

    SERIES = ('tip_x', 'V_in', 'N_reach', 'X_eat')

    def __init__(self, model, params, x_coords, S0):
        super().__init__(model, params, x_coords, S0)
        self.D = float(params.get('D_val') or 1.0)
        self.f_abi = float(params.get('f_abi') or 0.0)
        self.x_b = float(params['L_naive'])
        self.x_e = self.x_b + float(params['L_primed'])
        self.has_after = float(params.get('L_after') or 0.0) > 0
        self.abortive_sub = (model.get_abortive_subspace(),)
        # Node interval holding the entrance, for the flux
        self.i_b = int(np.clip(
            np.searchsorted(self.x, self.x_b, side='right') - 1,
            0, len(self.x) - 2,
        ))
        self.in_barrier = (self.x > self.x_b) & (self.x <= self.x_e)
        self.t_prev = None
        self.flux_prev = 0.0
        self.N = 0.0
        self.t_breach = np.nan

    def update(self, t, field):
        super().update(t, field)
        V = field((0,))
        i = self.i_b
        flux = -self.D * (V[i + 1] - V[i]) / (self.x[i + 1] - self.x[i])
        if self.t_prev is not None:
            # Trapezoidal rule in time
            self.N += 0.5 * (flux + self.flux_prev) * (t - self.t_prev)
        self.t_prev, self.flux_prev = t, flux

        X_eat = np.nan
        if self.f_abi > 0:
            R = field(self.abortive_sub)[self.in_barrier]
            edge = leading_edge(
                self.x[self.in_barrier], -R, -0.5 * self.f_abi * self.S0
            )
            X_eat = 0.0 if np.isnan(edge) else max(edge - self.x_b, 0.0)

        tip = self.series['tip_x'][-1]
        if self.has_after and np.isnan(self.t_breach) and tip > self.x_e:
            self.t_breach = t

        self.series['V_in'].append(float(np.interp(self.x_b, self.x, V)))
        self.series['N_reach'].append(self.N)
        self.series['X_eat'].append(X_eat)

    def summary(self):
        def last(name):
            vals = np.asarray(self.series[name], dtype=float)
            vals = vals[~np.isnan(vals)]
            return float(vals[-1]) if len(vals) else np.nan

        V_in = np.asarray(self.series['V_in'], dtype=float)
        return {
            'V_in_max': float(np.nanmax(V_in)) if len(V_in) else np.nan,
            'N_reach_final': last('N_reach'),
            'X_eat_final': last('X_eat'),
            't_breach': self.t_breach,
        }


class SignalFrontTracker(FrontTracker):
    """Tracks tip, primed band and signal metrics every time step.

    Needs the model to provide get_front_subspaces(),
    get_protein_subspace(), resistant_share(p), get_signal_subspace() and
    signal_threshold(params). The primed (resistant) uninfected density
    is R = phi(p) U, with phi the model's Michaelis-Menten response.
    """

    SUMMARY_COLUMNS = SUMMARY_COLUMNS

    SERIES = (
        'tip_x', 'primed_edge_x', 'primed_band_width', 'L_halo',
        'L_halo_mid', 'sigma_tip', 'x_K', 'z_K',
    )

    def __init__(self, model, params, x_coords, S0):
        super().__init__(model, params, x_coords, S0)
        self.K = float(model.signal_threshold(params))
        self.protein_sub = (model.get_protein_subspace(),)
        self.resistant_share = model.resistant_share
        # Models with an explicit resistant compartment (finite switching)
        get_R = getattr(model, 'get_resistant_subspace', None)
        self.resistant_sub = (get_R(),) if get_R is not None else None
        self.signal_sub = (model.get_signal_subspace(),)
        self.primed_edge_snap = []
        self.t_signal_x = np.full(len(self.x), np.nan)
        self.t_tip_x = np.full(len(self.x), np.nan)

    def primed_edge(self, R):
        return leading_edge(self.x, R, PRIMED_LEVEL * self.S0)

    def resistant_density(self, field):
        """(U, R, phi): uninfected, resistant uninfected R = phi(p) U and
        the resistant share phi(p). With an explicit R compartment, R is
        read directly and phi is the actual share R / U."""
        U = field(self.front_subs)
        if self.resistant_sub is not None:
            R = field(self.resistant_sub)
            share = np.divide(R, U, out=np.zeros_like(R), where=U > 1e-12)
            return U, R, share
        phi = self.resistant_share(field(self.protein_sub))
        return U, phi * U, phi

    def update(self, t, field):
        U, R, phi = self.resistant_density(field)
        self._update(t, U, R, field(self.signal_sub), phi)

    def _update(self, t, U, R, F, phi):
        x = self.x
        tip = self.tip(U)
        mid = leading_edge(x, -U, -0.5 * self.S0)
        phi_edge = leading_edge(x, phi, 0.5)
        if np.isnan(mid):
            halo_mid = np.nan
        elif np.isnan(phi_edge):
            halo_mid = 0.0
        else:
            halo_mid = max(phi_edge - mid, 0.0)
        primed_mask = R >= PRIMED_LEVEL * self.S0
        edge = self.primed_edge(R)
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
            'L_halo_mid': halo_mid,
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
            self.primed_edge(self.resistant_density(field)[1])
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
            'L_halo_mid_max': peak('L_halo_mid'),
            'L_halo_mid_final': last('L_halo_mid'),
            'primed_band_width_max': peak('primed_band_width'),
            'primed_band_width_final': last('primed_band_width'),
            'sigma_tip_final': last('sigma_tip'),
            'z_K_final': last('z_K'),
            'warning_median': (
                float(np.median(warn[keep])) if np.any(keep) else np.nan
            ),
        }
        return {key: summary[key] for key in SUMMARY_COLUMNS}
