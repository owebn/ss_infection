"""Front regime classification from a finished run.

Works on any SimulationResult (live or loaded from a history file), using
only the per-step series every model records: ``t``, ``tip_x`` (leading
infection edge, metrics.FrontTracker), ``front_x``, plus the final phage
profile and, for signal models, ``x_K`` and ``L_halo``.

Speeds come from a sliding-window least-squares fit of ``tip_x`` against
time, not from point differences, so grid steps in the front position do
not show up as noise.

Regimes, checked in this order:
    failed         the solver did not converge
    undetermined   too little front data after the transient
    extinct        free phage below V_ext (p_ext) everywhere at the end
    silent         signal model whose signal never reaches K after the
                   transient (the front runs as in a naive lawn)
    stop_and_go    >= 2 stalls that each ended with a restart
    stall          the front is stalled at the end with phage still alive
                   (arrest, possibly followed by slow creep)
    single_stall   one stall with a restart, not stalled at the end
    wave           steady speed (non-signal model)
    wave_fast      steady speed, no primed halo ahead of the tip
    wave_halo      steady speed with a primed halo ahead of the tip
    unsteady       none of the above

A stall is a stretch of at least ``stall_min`` during which the smoothed
tip speed stays below ``stall_frac * c_ref``. The reference speed
``c_ref`` is the dispersion-relation speed in a fully susceptible lawn.

Thresholds (params keys; None or missing = default):
    regime_transient   time skipped after the bolus
                       (default max(3 tau, 10% of run), at most 50%)
    regime_window      speed-fit window (default max(tau, 20 dt))
    stall_frac         stall speed threshold / c_ref (default 0.25; the
                       slow phases of manuscript set B run at ~0.14 c_ref,
                       so 0.1 misses them)
    stall_min          minimum stall duration (default regime_window)
    cv_steady          max speed coefficient of variation for a wave
                       (default 0.15)
    halo_tol           L_halo above which a wave carries a halo
                       (default 1.0)
"""

from dataclasses import dataclass, field

import numpy as np

# Per-run regime metrics, in CSV column order
REGIME_COLUMNS = (
    'regime', 'c_ref', 'c_mean', 'c_cv', 'X_max', 't_arrest',
    'n_stalls', 'T_cycle', 'L_step', 'hit_domain_end',
)


@dataclass
class RegimeSettings:
    transient: float
    window: float
    stall_frac: float = 0.25
    stall_min: float = None
    cv_steady: float = 0.15
    halo_tol: float = 1.0

    @classmethod
    def from_params(cls, params, t_last):
        def get(key, default):
            val = params.get(key)
            return default if val is None else float(val)

        tau = float(params.get('tau', 1.0))
        dt = float(params.get('dt', 0.1))
        transient = get(
            'regime_transient', min(max(3.0 * tau, 0.1 * t_last), 0.5 * t_last)
        )
        window = get('regime_window', max(tau, 20.0 * dt))
        return cls(
            transient=transient,
            window=window,
            stall_frac=get('stall_frac', 0.25),
            stall_min=get('stall_min', window),
            cv_steady=get('cv_steady', 0.15),
            halo_tol=get('halo_tol', 1.0),
        )


@dataclass
class RegimeAnalysis:
    """Classification plus the per-step quantities behind it."""

    regime: str
    speed: np.ndarray          # smoothed tip speed per step (NaN = no tip)
    stall_mask: np.ndarray     # 1.0 during stalls, else 0.0
    stalls: list               # [(t_start, t_end), ...]
    settings: RegimeSettings
    metrics: dict = field(default_factory=dict)

    def series(self):
        """Per-step arrays to store with the run (aligned with ``t``)."""
        return {'c_tip': self.speed, 'stall': self.stall_mask}


def sliding_speed(t, x, window):
    """Least-squares slope of x(t) in a centred window of width ``window``.

    Points where x is NaN are ignored; windows are clipped at the ends of
    the record. Returns NaN where x is NaN or fewer than 3 points fit.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = ~np.isnan(x)
    tc = t - (t.mean() if len(t) else 0.0)
    w = ok.astype(float)
    tz = np.where(ok, tc, 0.0)
    xz = np.where(ok, x, 0.0)

    def csum(a):
        return np.concatenate(([0.0], np.cumsum(a)))

    S1, St, Sx = csum(w), csum(tz), csum(xz)
    Stt, Stx = csum(tz * tz), csum(tz * xz)

    lo = np.searchsorted(t, t - window / 2.0, side='left')
    hi = np.searchsorted(t, t + window / 2.0, side='right')
    n = S1[hi] - S1[lo]
    st, sx = St[hi] - St[lo], Sx[hi] - Sx[lo]
    stt, stx = Stt[hi] - Stt[lo], Stx[hi] - Stx[lo]
    denom = n * stt - st * st
    with np.errstate(invalid='ignore', divide='ignore'):
        slope = (n * stx - st * sx) / denom
    slope[(n < 3) | (denom <= 0) | ~ok] = np.nan
    return slope


def find_stalls(t, speed, threshold, min_duration, t_start):
    """Intervals after ``t_start`` where speed < threshold for long enough.

    NaN speeds (no tip yet) never count as stalls.
    """
    below = (speed < threshold) & (t >= t_start)
    below &= ~np.isnan(speed)
    stalls = []
    i, n = 0, len(t)
    while i < n:
        if below[i]:
            j = i
            while j + 1 < n and below[j + 1]:
                j += 1
            if t[j] - t[i] >= min_duration:
                stalls.append((float(t[i]), float(t[j])))
            i = j + 1
        else:
            i += 1
    return stalls


def reference_speed(model, params):
    """Leading-edge speed in a fully susceptible lawn (f_abi = 0)."""
    try:
        c_ref, _ = type(model).theoretical_speed({**params, 'f_abi': 0.0})
        return float(c_ref)
    except Exception:
        return float('nan')


def analyze(result, **overrides):
    """Classify the front regime of a SimulationResult.

    Keyword overrides replace threshold params (see module docstring).
    """
    params = {**result.params, **overrides}
    t = np.asarray(result.series.get('t', []), dtype=float)
    tip = np.asarray(result.series.get('tip_x', np.full(len(t), np.nan)),
                     dtype=float)
    t_last = float(t[-1]) if len(t) else 0.0
    settings = RegimeSettings.from_params(params, t_last)

    speed = sliding_speed(t, tip, settings.window) if len(t) else t
    c_ref = reference_speed(result.model, params)
    if not np.isfinite(c_ref) or c_ref <= 0:
        after = speed[t >= settings.transient]
        c_ref = float(np.nanmax(after)) if np.any(~np.isnan(after)) \
            else float('nan')
    threshold = settings.stall_frac * c_ref if np.isfinite(c_ref) else 0.0
    stalls = find_stalls(t, speed, threshold, settings.stall_min,
                         settings.transient)
    stall_mask = np.zeros(len(t))
    for t0, t1 in stalls:
        stall_mask[(t >= t0) & (t <= t1)] = 1.0

    late = t >= settings.transient
    tip_late = tip[late]
    tip_ok = ~np.isnan(tip_late)
    eps = 0.5 * float(params.get('dt', 0.1))

    # Net advance over the analysed window (the review's c-bar)
    if np.count_nonzero(tip_ok) >= 2:
        t_ok, x_ok = t[late][tip_ok], tip_late[tip_ok]
        c_mean = float((x_ok[-1] - x_ok[0]) / max(t_ok[-1] - t_ok[0], eps))
    else:
        c_mean = float('nan')
    spd = speed[late]
    spd = spd[~np.isnan(spd)]
    c_cv = (float(np.std(spd) / np.mean(spd))
            if len(spd) and np.mean(spd) > 0 else float('nan'))

    final_stall = bool(stalls) and stalls[-1][1] >= t_last - eps
    restarts = stalls[:-1] if final_stall else stalls
    starts = [s[0] for s in stalls]
    tips_at = [float(np.interp(s, t, np.nan_to_num(tip))) for s in starts]

    front_x = np.asarray(result.series.get('front_x', []), dtype=float)
    hit_end = float(len(front_x) > 0
                    and front_x[-1] >= 0.8 * float(params['L']))

    metrics = {
        'c_ref': c_ref,
        'c_mean': c_mean,
        'c_cv': c_cv,
        'X_max': float(np.nanmax(tip)) if np.any(~np.isnan(tip))
        else float('nan'),
        't_arrest': stalls[-1][0] if final_stall else float('nan'),
        'n_stalls': float(len(stalls)),
        'T_cycle': float(np.mean(np.diff(starts))) if len(starts) >= 2
        else float('nan'),
        'L_step': float(np.mean(np.diff(tips_at))) if len(starts) >= 2
        else float('nan'),
        'hit_domain_end': hit_end,
    }

    regime = _classify(
        result, params, settings, t, late, tip_ok, c_cv, final_stall,
        restarts,
    )
    metrics = {'regime': regime, **metrics}
    return RegimeAnalysis(
        regime=regime, speed=speed, stall_mask=stall_mask, stalls=stalls,
        settings=settings,
        metrics={col: metrics[col] for col in REGIME_COLUMNS},
    )


def _classify(result, params, settings, t, late, tip_ok, c_cv, final_stall,
              restarts):
    if not result.success:
        return 'failed'
    if np.count_nonzero(tip_ok) < 3:
        return 'undetermined'

    p_ext = float(params.get('p_ext', 1e-6))
    V = result.profiles.get('V')
    if V is not None and len(V) and float(np.max(V[-1])) < p_ext:
        return 'extinct'

    has_signal = hasattr(result.model, 'get_signal_subspace')
    if has_signal:
        x_K = np.asarray(result.series.get('x_K', []), dtype=float)
        if len(x_K) == len(t) and np.all(np.isnan(x_K[late])):
            return 'silent'

    if len(restarts) >= 2:
        return 'stop_and_go'
    if final_stall:
        return 'stall'
    if len(restarts) == 1:
        return 'single_stall'
    if np.isfinite(c_cv) and c_cv < settings.cv_steady:
        if not has_signal:
            return 'wave'
        halo = np.asarray(result.series.get('L_halo', []), dtype=float)
        halo = halo[late] if len(halo) == len(t) else halo
        halo = halo[~np.isnan(halo)]
        if len(halo) and float(np.median(halo)) > settings.halo_tol:
            return 'wave_halo'
        return 'wave_fast'
    return 'unsteady'
