"""RNA-protein host response: priming curve, ceiling and reversion.

Closed forms of the first review, revision of Oct 24 2026 (§§ 3.1-3.6):

    dm/dt = alpha_m H(F) - delta_m m,    dp/dt = alpha_p m - delta_p p,
    phi(p) = phi_max p / (K_p + p)       (resistant proportion)

For a bin whose signal stays above threshold from t = 0 (h >> 1), starting
from m = p = 0,

    p(t) = p_inf [1 - (delta_p e^{-delta_m t} - delta_m e^{-delta_p t})
                       / (delta_p - delta_m)],   p_inf = alpha_m alpha_p
                                                         / (delta_m delta_p)

(for delta_m = delta_p = delta: p_inf [1 - (1 + delta t) e^{-delta t}]),
and the priming curve is Phi(t) = phi(p(t)). From saturation, the decay
after the signal clears is the complement p_inf - p(t).

A ``response_timescale`` tau_R divides all four rates, which stretches
every time by tau_R and leaves p_inf and phi_inf unchanged.
"""

import math
from dataclasses import dataclass

from scipy.optimize import brentq


@dataclass(frozen=True)
class HostResponse:
    alpha_m: float = 1.0
    delta_m: float = 1.0
    alpha_p: float = 2.0
    delta_p: float = 0.1
    phi_max: float = 1.0
    K_p: float = 1.0

    @classmethod
    def from_params(cls, params):
        """Rates from a params dict, scaled by ``response_timescale``."""
        def get(key, default):
            val = params.get(key)
            return float(default if val is None else val)

        scale = get('response_timescale', 1.0)
        return cls(
            alpha_m=get('alpha_m', 1.0) / scale,
            delta_m=get('delta_m', 1.0) / scale,
            alpha_p=get('alpha_p', 2.0) / scale,
            delta_p=get('delta_p', 0.1) / scale,
            phi_max=get('phi_max', 1.0),
            K_p=get('K_p', 1.0),
        )

    # -- steady states -------------------------------------------------------

    @property
    def p_inf(self):
        """Saturating protein alpha_m alpha_p / (delta_m delta_p)."""
        if self.delta_m <= 0 or self.delta_p <= 0:
            return math.inf
        return self.alpha_m * self.alpha_p / (self.delta_m * self.delta_p)

    def phi(self, p):
        return self.phi_max * p / (self.K_p + p)

    @property
    def phi_inf(self):
        """Resistance ceiling phi(p_inf) < phi_max."""
        if math.isinf(self.p_inf):
            return self.phi_max
        return self.phi(self.p_inf)

    def p_for(self, f):
        """Protein level giving resistant share f (inf if f >= phi_max)."""
        if f >= self.phi_max:
            return math.inf
        return self.K_p * f / (self.phi_max - f)

    # -- priming curve -----------------------------------------------------

    def protein(self, t):
        """p(t) after t above threshold, from m = p = 0."""
        dm, dp = self.delta_m, self.delta_p
        if t <= 0:
            return 0.0
        if dp == 0:
            # permanent protein: p grows without bound
            if dm == 0:
                return self.alpha_m * self.alpha_p * t * t / 2.0
            return self.alpha_m * self.alpha_p / dm * (
                t - (1.0 - math.exp(-dm * t)) / dm
            )
        if abs(dp - dm) < 1e-12 * max(dp, dm):
            return self.p_inf * (1.0 - (1.0 + dm * t) * math.exp(-dm * t))
        return self.p_inf * (
            1.0 - (dp * math.exp(-dm * t) - dm * math.exp(-dp * t)) / (dp - dm)
        )

    def priming_curve(self, t):
        """Phi(t): resistant share after t above threshold."""
        return self.phi(self.protein(t))

    def time_to_share(self, f):
        """t_f with Phi(t_f) = f, or inf if f is above the ceiling."""
        if f <= 0:
            return 0.0
        target = self.p_for(f)
        if not target < self.p_inf:
            return math.inf
        hi = 1.0
        while self.protein(hi) < target:
            hi *= 2.0
            if hi > 1e9:
                return math.inf
        return brentq(lambda t: self.protein(t) - target, 0.0, hi, xtol=1e-10)

    def reversion_time(self, f):
        """t_rev: time for resistance to fall below f after the signal
        clears from saturation, i.e. p_inf - p(t_rev) = p_f (eq. 30)."""
        p_f = self.p_for(f)
        if not p_f < self.p_inf:
            return 0.0
        if math.isinf(self.p_inf):
            return math.inf
        target = self.p_inf - p_f
        hi = 1.0
        while self.protein(hi) < target:
            hi *= 2.0
        return brentq(lambda t: self.protein(t) - target, 0.0, hi, xtol=1e-10)


def critical_fraction(b, b_R):
    """f_abi^c = (b - 1) / (b - b_R): resistant share that stops a front."""
    return (b - 1.0) / (b - b_R)


def summary(params):
    """Priming-curve quantities for --theoretical printouts (§ 8.2)."""
    hr = HostResponse.from_params(params)
    b = float(params.get('b_val') or 5.0)
    b_R = float(params.get('b_abi') or 0.0)
    f_c = critical_fraction(b, b_R)
    b_eff_inf = b - (b - b_R) * hr.phi_inf
    return {
        'p_inf': hr.p_inf,
        'phi_inf': hr.phi_inf,
        't_half': hr.time_to_share(0.5),
        't_star': hr.time_to_share(f_c),
        't_rev': hr.reversion_time(f_c),
        'f_c': f_c,
        'b_eff_inf': b_eff_inf,
        'stall_possible': b_eff_inf < 1.0,
    }
