import abc
import math

import numpy as np

from . import extinction, host_response
from .dispersion import dispersion_args, solve_dispersion_erlang
from .host_response import HostResponse
from .initial_conditions import (
    AbiBolusInitialCondition,
    AbiHeterogeneousInitialCondition,
    BolusInitialCondition,
    ImmuneSignalingInitialCondition,
)
from .metrics import FrontTracker, SignalFrontTracker
from .outputs import Overlay, ProfileGroup, SeriesSpec
from .regimes import REGIME_COLUMNS

try:
    import ufl
    from dolfinx import default_scalar_type, fem
except ImportError:
    ufl = None
    default_scalar_type = None
    fem = None


# Result columns every run reports (theory and simulation)
BASE_OUTPUT_COLUMNS = ('c_star', 'extinction', 'p_end', 't_extinction')


class BaseModel(abc.ABC):
    """Abstract base class for PDE models.

    Besides the PDE itself, a model declares everything the input and
    output code needs, so that adding a model means adding one class to
    MODEL_CLASSES (bottom of this module) and nothing else:

        name, aliases        value(s) of params['model'] / --model
        initial_condition    BaseInitialCondition subclass to apply
        param_aliases        sweep parameter aliases (alias -> canonical)
        sweep_defaults       sweep parameters filled in when not given
        implied_by           sweep parameters that select this model when
                             the sweep names none (see factory.infer_model)
        from_params()        construct the model from a params dict
        prepare_params()     derived parameters (w_val, k_abi, L, ...)
        adaptive_overrides() adaptive sweep resolution (L, dx, t_end, dt)
        theoretical_speed()  dispersion-relation speed (c*, lambda*)
        theoretical_extinction()  analytical (extinction, p_end, t_ext)
        describe_theory()    extra lines for --theoretical printouts
        profile_groups()     profiles to cache and draw in snapshots
        trackers             metric trackers (metrics.py) to run; their
                             SUMMARY_COLUMNS become CSV columns (regime
                             columns, regimes.py, are added for every run)
        speed_panel_series() extra series drawn on the speed-over-time
                             plot (e.g. L_halo on a secondary axis)
        reports_wave_speed   whether c* is meaningful (title, fit, print)
        kymograph_overlays() lines drawn on the kymograph
    """

    name = None
    aliases = ()
    initial_condition = None
    # History profile keys written before the manuscript notation
    # (old name -> current name); see SimulationResult.from_history.
    legacy_profile_names = {'P': 'V', 'B': 'S'}
    reports_wave_speed = True
    trackers = (FrontTracker,)
    has_lysogens = False

    # Subclasses extend these dicts ({**Parent.param_aliases, ...}).
    # Aliases of every model are applied before the model is known.
    param_aliases = {
        'b': 'b_val', 'burst': 'b_val', 'burst_size': 'b_val',
        'D': 'D_val', 'diffusion': 'D_val',
        'f_B': 'f_B_val', 'f_b': 'f_B_val', 'adsorption': 'f_B_val',
        'B0': 'B0_val', 'b0': 'B0_val', 'lawn': 'B0_val',
        'alpha': 'f_abi', 'f_abi_val': 'f_abi', 'abi_fraction': 'f_abi',
        'n': 'n_stages', 'stages': 'n_stages',
        'latent_period': 'tau', 'delay': 'tau',
        'P_extinction': 'p_ext', 'p_extinction': 'p_ext',
        # Manuscript notation (V phage, S naive, R resistant)
        'beta': 'f_B_val', 'beta_I': 'f_I_val', 'beta_L': 'f_L_val',
        'S0': 'B0_val', 'b_R': 'b_abi', 'delta_V': 'delta_P',
        'V_ext': 'p_ext',
    }
    # f_abi is filled for every model: the default dispersion relation
    # reads it (see theoretical_speed).
    sweep_defaults = {
        'tau': 5.0, 'n_stages': 10, 'k_abi': None, 'f_abi': 0.5,
        'D_val': 1.0, 'f_B_val': 1.0, 'b_val': 5.0,
        'B0_val': 1.0, 'dt': 0.1, 't_end': 25.0, 'dx': 0.1, 'L': 200.0,
        'p_ext': 1e-6,
    }
    implied_by = ()

    # -- construction / inputs ---------------------------------------------

    @classmethod
    def from_params(cls, params):
        return cls(n_stages=int(params['n_stages']))

    @classmethod
    def prepare_params(cls, params):
        """Fill derived parameters in place before the model is built.

        Also resolves parameter aliases (e.g. manuscript names such as
        ``beta`` or ``S0``) to their canonical keys, so every consumer
        (weak form, initial condition, trackers) reads the same value.
        """
        for alias, canonical in cls.param_aliases.items():
            if alias in params and params.get(canonical) is None:
                params[canonical] = params[alias]
        if params.get('k_abi') is None:
            params['k_abi'] = params['n_stages']
        params['w_val'] = float(params['n_stages']) / float(params['tau'])

    @classmethod
    def adaptive_overrides(cls, params):
        """Adaptive sweep resolution: {'L', 'dx', 't_end', 'dt'}.

        Uses the Fisher-KPP wave speed and front width estimates.
        """
        D = params.get('D_val', 1.0)
        f_B = params.get('f_B_val', 1.0)
        B0 = params.get('B0_val', 1.0)
        b = params.get('b_val', 5.0)

        k = f_B * B0
        if b > 1.0:
            c_est = 2.0 * math.sqrt(D * k * (b - 1.0))
            w_est = math.sqrt(D / k)
        else:
            c_est = 0.0
            w_est = 1.0

        L = max(200.0, 30.0 * w_est)
        dx = min(params.get('dx', 0.1), w_est / 10.0)
        t_end = params.get('t_end', 25.0)
        dt = params.get('dt', 0.1)
        if c_est > 0.0:
            t_end = max(25.0, 1.5 * ((0.6 * L) / c_est))
            dt = min(dt, 0.5 * dx / c_est)
        return {'L': L, 'dx': dx, 't_end': t_end, 'dt': dt}

    # -- theory --------------------------------------------------------------

    @classmethod
    def dispersion_args(cls, params):
        """Keyword arguments for dispersion.solve_dispersion_erlang."""
        return dispersion_args(params)

    @classmethod
    def theoretical_speed(cls, params):
        """Leading-edge wave speed: (c_star, lambda_star)."""
        return solve_dispersion_erlang(**cls.dispersion_args(params))

    @classmethod
    def theoretical_extinction(cls, params):
        """Analytical (extinction, p_end, t_extinction)."""
        scales = extinction.edge_scales(cls, params)
        if not isinstance(scales, extinction.EdgeScales):
            return scales
        return cls.extinction_from_scales(params, scales)

    @classmethod
    def extinction_from_scales(cls, params, scales):
        """Model-specific part of theoretical_extinction (NaN if none)."""
        return extinction.NAN_METRICS

    @classmethod
    def describe_theory(cls, params):
        """Model-specific lines for the --theoretical printout."""
        return []

    @classmethod
    def output_columns(cls, theoretical=False):
        """Result columns this model reports, in CSV order."""
        columns = list(BASE_OUTPUT_COLUMNS)
        if not theoretical:
            columns += list(REGIME_COLUMNS)
            for tracker in cls.trackers:
                columns += [
                    c for c in tracker.SUMMARY_COLUMNS if c not in columns
                ]
        return columns

    # -- PDE -----------------------------------------------------------------

    @property
    @abc.abstractmethod
    def num_components(self) -> int:
        """Return the number of components in the mixed space."""
        pass

    @abc.abstractmethod
    def create_weak_formulation(self, domain, W, u, u_n, params):
        """Define the weak formulation (UFL) of the PDE system.

        ``W`` is the mixed function space (``V`` is reserved for free
        phage, following the manuscript's notation).
        """
        pass

    def get_front_subspaces(self):
        """Return subspace indices used for wavefront tracking.

        The simulation sums the values from these subspaces and
        detects the front where the total drops below half the
        initial density.

        Returns:
            List of integer subspace indices.
        """
        return [1]

    @staticmethod
    def _phage_cutoff(V, params, domain):
        """Optional density cutoff (manuscript Methods; first review F1).

        With ``params['cutoff']`` true, phage below V_ext (``p_ext``)
        neither adsorb nor infect and are removed at ``cutoff_rate``
        (default 5), representing phage discreteness independently of the
        time step.

        Returns (V_ads, removal): the phage density used in every
        adsorption term, and the removal flux to subtract from dV/dt
        (None when the cutoff is off, in which case V_ads is V itself
        and the weak form is unchanged).
        """
        if not _param(params, ['cutoff'], False):
            return V, None
        V_ext = float(_param(
            params, ['p_ext', 'P_extinction', 'p_extinction', 'V_ext'], 1e-6
        ))
        rate = float(_param(params, ['cutoff_rate'], 5.0))
        V_ext_c = fem.Constant(domain, default_scalar_type(V_ext))
        rate_c = fem.Constant(domain, default_scalar_type(rate))
        above = ufl.conditional(ufl.ge(V, V_ext_c), 1.0, 0.0)
        return above * V, rate_c * (1.0 - above) * V

    # -- outputs -------------------------------------------------------------

    @abc.abstractmethod
    def profile_groups(self):
        """Return a list of outputs.ProfileGroup, in snapshot plot order."""
        pass

    def history_groups(self):
        """{name: subspace indices} for every cached profile.

        Always includes 'U', the uninfected total that defines the front
        and fills the kymograph.
        """
        groups = {'U': tuple(self.get_front_subspaces())}
        for group in self.profile_groups():
            if group.indices:
                groups[group.name] = tuple(group.indices)
        return groups

    def derived_profiles(self, snapshot):
        """Profiles computed from cached ones ({name: array}).

        Declared in profile_groups() with empty ``indices``; called on each
        snapshot after the component sums of history_groups() are taken.
        """
        return {}

    def kymograph_title(self, result):
        title = "Bacterial Density Spatiotemporal Dynamics"
        if self.reports_wave_speed:
            title += f" ($c^* = {result.metrics['c_star']:.4f}$)"
        return title

    def speed_panel_series(self):
        """Extra series for the speed plot (list of outputs.SeriesSpec)."""
        return []

    def kymograph_overlays(self, result):
        """Lines to draw on the kymograph (list of outputs.Overlay)."""
        overlays = []
        fit_t = result.series.get('fit_t', [])
        if self.reports_wave_speed and len(fit_t) > 0:
            overlays.append(Overlay(
                'line',
                f"Linear Fit ($c^* = {result.metrics['c_star']:.4f}$)",
                t=fit_t, x=result.series['fit_x'],
                style=dict(color='red', linestyle='--'),
            ))
        return overlays


class DelayChainModel(BaseModel):
    """Delay-chain (Erlang) bacteriophage infection model (manuscript § 1.1).

    Free phage V diffuse (D) and adsorb to susceptible cells S at rate
    beta; infected cells pass through latent stages I_1 .. I_n at rate w
    and lyse from I_n releasing b phage:

        dV/dt = D V'' - beta V S - beta V sum(I_k) + b w I_n
        dS/dt = -beta V S
        dI_1/dt = beta V S - w I_1,  dI_k/dt = w (I_{k-1} - I_k)

    (Shielding adsorption onto infected cells uses beta here.)

    Component layout: 0: V, 1: S, 2 .. 2+n-1: I_1 .. I_n
    """

    name = 'delay_chain'
    initial_condition = BolusInitialCondition

    def __init__(self, n_stages: int):
        self.n_stages = n_stages

    @property
    def num_components(self) -> int:
        return 2 + self.n_stages

    def create_weak_formulation(
        self, domain, W, u, u_n, params
    ):
        D_val = params.get('D_val', params.get('D', 1.0))
        beta_val = params.get(
            'f_B_val', params.get('f_B', params.get('beta', 1.0))
        )
        w_val = params.get('w_val', 2.0)
        b_val = params.get('b_val', params.get('b', 5.0))
        dt_val = params.get('dt', 0.1)

        D_c = fem.Constant(domain, default_scalar_type(D_val))
        beta_c = fem.Constant(domain, default_scalar_type(beta_val))
        w_c = fem.Constant(domain, default_scalar_type(w_val))
        burst_c = fem.Constant(domain, default_scalar_type(b_val))
        dt_c = fem.Constant(domain, default_scalar_type(dt_val))

        v = ufl.TestFunctions(W)
        u_components = ufl.split(u)
        u_n_components = ufl.split(u_n)

        V, S = u_components[0], u_components[1]
        I_ = u_components[2:]
        V_n, S_n = u_n_components[0], u_n_components[1]
        I_n = u_n_components[2:]

        v_V, v_S = v[0], v[1]
        v_I = v[2:]
        dx = ufl.dx
        V_ads, removal = self._phage_cutoff(V, params, domain)

        eq_V = (
            ((V - V_n) / dt_c) * v_V * dx
            + D_c * ufl.inner(
                ufl.grad(V), ufl.grad(v_V)
            ) * dx
            - (
                -beta_c * V_ads * S
                - beta_c * V_ads * sum(I_)
                + burst_c * w_c * I_[-1]
            ) * v_V * dx
        )

        if removal is not None:
            eq_V += removal * v_V * dx

        eq_S = (
            ((S - S_n) / dt_c) * v_S * dx
            - (-beta_c * V_ads * S) * v_S * dx
        )

        eq_I_list = []
        eq_I1 = (
            ((I_[0] - I_n[0]) / dt_c) * v_I[0] * dx
            - (beta_c * V_ads * S - w_c * I_[0]) * v_I[0] * dx
        )
        eq_I_list.append(eq_I1)

        for k in range(1, self.n_stages):
            eq_Ik = (
                ((I_[k] - I_n[k]) / dt_c) * v_I[k] * dx
                - (
                    w_c * I_[k - 1] - w_c * I_[k]
                ) * v_I[k] * dx
            )
            eq_I_list.append(eq_Ik)

        return eq_V + eq_S + sum(eq_I_list)

    def profile_groups(self):
        return [
            ProfileGroup('S', (1,), r'$S$ (susceptible)', 'orange'),
            ProfileGroup('V', (0,), r'$V$ (phage)', 'purple', is_phage=True),
            ProfileGroup(
                'I', tuple(range(2, 2 + self.n_stages)),
                r'$I = \sum I_k$', 'teal',
            ),
        ]


class AbortiveInfectionModel(BaseModel):
    """Delay-chain model with abortive cells placed in the lawn (§ 2.1).

    A fraction f_abi of the lawn starts in the abortive (primed) state,
    written R (the resistant state of the signalling models). They adsorb phage
    at the normal rate beta, and their infections (J_1 .. J_k_abi) die
    without releasing phage (b_R = 0 here).

    Component layout in the mixed function space:
        0:  V   (free phage)
        1:  S   (susceptible bacteria)
        2:  R   (abortive bacteria)
        3 .. 3+n-1:        I_1 .. I_n  (productive chain)
        3+n .. 3+n+k_abi-1: J_1 .. J_k  (abortive chain)
    """

    name = 'abi'
    initial_condition = AbiBolusInitialCondition
    legacy_profile_names = {'P': 'V', 'B_S': 'S', 'B_A': 'R'}
    implied_by = ('f_abi', 'k_abi')

    def __init__(self, n_stages: int, k_abi: int):
        self.n_stages = n_stages
        self.k_abi = k_abi

    @property
    def num_components(self) -> int:
        return 3 + self.n_stages + self.k_abi

    def get_front_subspaces(self):
        """Track depletion of all uninfected cells, S + R."""
        return [1, 2]

    def create_weak_formulation(
        self, domain, W, u, u_n, params
    ):
        D_val = params.get('D_val', params.get('D', 1.0))
        beta_val = params.get(
            'f_B_val', params.get('f_B', params.get('beta', 1.0))
        )
        w_val = params.get('w_val', 2.0)
        b_val = params.get('b_val', params.get('b', 5.0))
        dt_val = params.get('dt', 0.1)

        D_c = fem.Constant(domain, default_scalar_type(D_val))
        beta_c = fem.Constant(domain, default_scalar_type(beta_val))
        w_c = fem.Constant(domain, default_scalar_type(w_val))
        burst_c = fem.Constant(domain, default_scalar_type(b_val))
        dt_c = fem.Constant(domain, default_scalar_type(dt_val))

        v = ufl.TestFunctions(W)
        u_comp = ufl.split(u)
        u_n_comp = ufl.split(u_n)

        # Current time step
        V = u_comp[0]
        S = u_comp[1]
        R = u_comp[2]
        i_start = 3
        i_end = i_start + self.n_stages
        I_ = u_comp[i_start:i_end]
        j_start = i_end
        j_end = j_start + self.k_abi
        J_ = u_comp[j_start:j_end]

        # Previous time step
        V_n = u_n_comp[0]
        S_n = u_n_comp[1]
        R_n = u_n_comp[2]
        I_n = u_n_comp[i_start:i_end]
        J_n = u_n_comp[j_start:j_end]

        # Test functions
        v_V = v[0]
        v_S = v[1]
        v_R = v[2]
        v_I = v[i_start:i_end]
        v_J = v[j_start:j_end]

        dx = ufl.dx
        V_ads, removal = self._phage_cutoff(V, params, domain)

        # --- Free phage (V) ---
        # Phage diffuses, is consumed by adsorption onto
        # both S and R (at rate beta), is unproductively
        # adsorbed onto infected cells (at rate beta), and is
        # produced only by lysis of the productive chain.
        all_infected = list(I_) + list(J_)
        eq_V = (
            ((V - V_n) / dt_c) * v_V * dx
            + D_c * ufl.inner(
                ufl.grad(V), ufl.grad(v_V)
            ) * dx
            - (
                -beta_c * V_ads * S
                - beta_c * V_ads * R
                - beta_c * V_ads * sum(all_infected)
                + burst_c * w_c * I_[-1]
            ) * v_V * dx
        )

        if removal is not None:
            eq_V += removal * v_V * dx

        # --- Susceptible cells (S) ---
        eq_S = (
            ((S - S_n) / dt_c) * v_S * dx
            - (-beta_c * V_ads * S) * v_S * dx
        )

        # --- Abortive cells (R) ---
        eq_R = (
            ((R - R_n) / dt_c) * v_R * dx
            - (-beta_c * V_ads * R) * v_R * dx
        )

        # --- Productive infected chain (I_1 .. I_n) ---
        eq_I_list = []
        eq_I1 = (
            ((I_[0] - I_n[0]) / dt_c) * v_I[0] * dx
            - (
                beta_c * V_ads * S - w_c * I_[0]
            ) * v_I[0] * dx
        )
        eq_I_list.append(eq_I1)

        for k in range(1, self.n_stages):
            eq_Ik = (
                ((I_[k] - I_n[k]) / dt_c)
                * v_I[k] * dx
                - (
                    w_c * I_[k - 1] - w_c * I_[k]
                ) * v_I[k] * dx
            )
            eq_I_list.append(eq_Ik)

        # --- Abortive infected chain (J_1 .. J_k_abi) ---
        eq_J_list = []
        eq_J1 = (
            ((J_[0] - J_n[0]) / dt_c) * v_J[0] * dx
            - (
                beta_c * V_ads * R - w_c * J_[0]
            ) * v_J[0] * dx
        )
        eq_J_list.append(eq_J1)

        for k in range(1, self.k_abi):
            eq_Jk = (
                ((J_[k] - J_n[k]) / dt_c)
                * v_J[k] * dx
                - (
                    w_c * J_[k - 1] - w_c * J_[k]
                ) * v_J[k] * dx
            )
            eq_J_list.append(eq_Jk)

        return (
            eq_V + eq_S + eq_R
            + sum(eq_I_list) + sum(eq_J_list)
        )

    @classmethod
    def from_params(cls, params):
        return cls(n_stages=int(params['n_stages']), k_abi=params['k_abi'])

    @classmethod
    def describe_theory(cls, params):
        return [
            f"Abi parameters: f_abi={params.get('f_abi')}, "
            f"k_abi={params.get('k_abi')}"
        ]

    def profile_groups(self):
        i_start = 3
        j_start = i_start + self.n_stages
        return [
            ProfileGroup('S', (1,), r'$S$ (susceptible)', 'orange'),
            ProfileGroup('R', (2,), r'$R$ (abortive)', 'red'),
            ProfileGroup('V', (0,), r'$V$ (phage)', 'purple', is_phage=True),
            ProfileGroup(
                'I', tuple(range(i_start, j_start)),
                r'$\sum I_k$ (infected)', 'teal',
            ),
            ProfileGroup(
                'J', tuple(range(j_start, j_start + self.k_abi)),
                r'$\sum J_k$ (abortive infected)', 'gray',
            ),
        ]


class AbiHeterogeneousModel(AbortiveInfectionModel):
    """PDE model for spatial heterogeneity (naive region + primed patch).

    Inherits the exact PDE weak formulation and component layout from
    AbortiveInfectionModel. Spatial heterogeneity is defined via initial
    bacterial lawn distributions.
    """

    name = 'abi_heterogeneous'
    initial_condition = AbiHeterogeneousInitialCondition
    param_aliases = {
        **AbortiveInfectionModel.param_aliases,
        'l_naive': 'L_naive', 'l_primed': 'L_primed',
    }
    sweep_defaults = {
        **AbortiveInfectionModel.sweep_defaults,
        'L_naive': 50.0, 'L_primed': 50.0,
    }
    implied_by = ('L_naive', 'L_primed')

    @classmethod
    def prepare_params(cls, params):
        super().prepare_params(params)
        params['L'] = float(params['L_naive']) + float(params['L_primed'])

    @staticmethod
    def _patch_lengths(params):
        return (
            float(params.get('L_naive', params.get('l_naive', 50.0))),
            float(params.get('L_primed', params.get('l_primed', 50.0))),
        )

    @classmethod
    def adaptive_overrides(cls, params):
        """Domain spans both patches; t_end covers crossing both."""
        D = params.get('D_val', 1.0)
        f_B = params.get('f_B_val', 1.0)
        B0 = params.get('B0_val', 1.0)
        L_naive, L_primed = cls._patch_lengths(params)
        L = L_naive + L_primed

        c_naive_est, _ = cls.theoretical_speed({**params, 'f_abi': 0.0})
        c_naive_est = max(0.1, c_naive_est)
        c_primed_est, _ = cls.theoretical_speed(params)
        c_eff_primed = max(c_primed_est, c_naive_est / 4.0)

        t_travel = (L_naive / c_naive_est) + (L_primed / c_eff_primed)
        w_est = math.sqrt(D / (f_B * B0)) if f_B * B0 > 0 else 1.0
        dx = min(params.get('dx', 0.1), w_est / 10.0)
        return {
            'L': L, 'dx': dx, 't_end': max(25.0, 1.5 * t_travel),
            'dt': min(params.get('dt', 0.1), 0.5 * dx / c_naive_est),
        }

    @classmethod
    def extinction_from_scales(cls, params, scales):
        """Phage dies out if it cannot cross the primed patch."""
        L_naive, L_primed = cls._patch_lengths(params)
        f_abi = float(
            params.get(
                'f_abi', params.get('f_abi_val', params.get('alpha', 0.5))
            )
        )

        if (1.0 - f_abi) * scales.b > 1.0:
            # Phage replicates in primed patch, no extinction
            return 0, scales.P_max, float('nan')
        p_end = scales.P_max * math.exp(-L_primed / scales.decay_length)
        if L_primed >= scales.L_primed_crit or p_end < scales.p_ext:
            t_ext = (L_naive + scales.L_primed_crit) / scales.c_eff
            return 1, p_end, t_ext
        return 0, p_end, float('nan')

    def kymograph_overlays(self, result):
        params = result.params
        L_naive = params.get('L_naive', params.get('l_naive', None))
        L_primed = params.get('L_primed', params.get('l_primed', None))
        x_max = float(np.max(result.x))
        overlays = []
        if L_naive is not None and float(L_naive) < x_max:
            overlays.append(Overlay(
                'hline', 'Naive / Primed Boundary', y=float(L_naive),
                style=dict(color='black', linestyle='--'),
            ))
            if L_primed is not None:
                L_primed_end = float(L_naive) + float(L_primed)
                if L_primed_end <= x_max:
                    overlays.append(Overlay(
                        'hline', 'End of Primed Patch', y=L_primed_end,
                        style=dict(color='black', linestyle='-'),
                    ))
        return overlays + super().kymograph_overlays(result)

# ---------------------------------------------------------------------------
# Lysogeny and signal-sensing models (notation of the first review, §§ 3-6)
# ---------------------------------------------------------------------------


def _param(params, keys, default):
    """Return the first non-None value among ``keys`` in ``params``.

    CLI parameter dictionaries carry ``None`` for optional arguments, so
    ``dict.get`` alone would not fall back to the default.
    """
    for key in keys:
        val = params.get(key, None)
        if val is not None:
            return val
    return default


def _lysogeny_window(params, n_stages):
    """Return the decision-window mask chi_k (1-based, inclusive bounds)."""
    start = int(_param(params, ['lys_window_start'], 1))
    end = int(_param(params, ['lys_window_end'], n_stages))
    start = max(1, start)
    end = min(n_stages, end)
    if start > end:
        raise ValueError(
            f"Empty lysogeny window: stages {start}..{end} of {n_stages}."
        )
    return [1.0 if start <= k + 1 <= end else 0.0 for k in range(n_stages)]


def lysogeny_hazard(p, w, n_window):
    """Per-stage hazard ell giving lysogeny probability p over a window.

    Inverts p = 1 - (w / (w + ell))**j (review eq. 62) for a window of
    j = n_window stages at per-stage latent rate w.
    """
    p = float(p)
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        raise ValueError("Lysogeny probability must be < 1.")
    return float(w) * ((1.0 - p) ** (-1.0 / n_window) - 1.0)


def _lysogeny_dispersion_args(params):
    """Dispersion inputs with leading-edge lysogeny hazards ell_k.

    These models have no static Abi fraction (f_abi = 0); induction is
    neglected at the leading edge.
    """
    args = dict(dispersion_args(params), f_abi=0.0)
    n = args['n']
    chi = _lysogeny_window(params, n)
    ell = lysogeny_hazard(
        _param(params, ['p_lys'], 0.0), args['w'], int(sum(chi))
    )
    args['hazards'] = [chi_k * ell for chi_k in chi]
    return args


def _describe_lysogeny(params):
    end = params.get('lys_window_end') or params['n_stages']
    return (
        f"Lysogeny parameters: p_lys={params.get('p_lys')}, window="
        f"{params.get('lys_window_start') or 1}..{end}"
    )


def _hill(F, K, h):
    """Hill switch F^h / (K^h + F^h), with F clipped at 0."""
    ratio = ufl.max_value(F, 0.0) / K
    return ratio**h / (1.0 + ratio**h)


class LysogenyDelayChainModel(BaseModel):
    """Delay-chain model for a temperate phage with constant lysogeny.

    Manuscript § 6.1 without the signal. Infected cells I_k divert to the
    lysogen compartment L at per-stage hazard ell_k = chi_k * ell, chosen
    so that a cell lysogenizes with total probability p_lys over the
    decision window. Lysogens adsorb phage at rate beta_L (phage lost)
    and induce into I_1 at rate nu; free phage decay at delta_V.

    Component layout in the mixed function space:
        0: V (free phage)
        1: S (susceptible cells)
        2 .. 2+n-1: I_1 .. I_n (latent chain)
        2+n: L (lysogens)
    """

    name = 'lysogeny_delay_chain'
    initial_condition = BolusInitialCondition
    has_lysogens = True

    def __init__(self, n_stages: int):
        self.n_stages = n_stages

    @property
    def num_components(self) -> int:
        return 3 + self.n_stages

    def lysogen_subspace(self):
        return 2 + self.n_stages

    def create_weak_formulation(self, domain, W, u, u_n, params):
        n = self.n_stages
        D_val = _param(params, ['D_val', 'D'], 1.0)
        beta_val = _param(params, ['f_B_val', 'f_B', 'beta'], 1.0)
        beta_I_val = _param(params, ['f_I_val', 'f_I', 'beta_I'], beta_val)
        beta_L_val = _param(params, ['f_L_val', 'f_L', 'beta_L'], beta_val)
        w_val = _param(params, ['w_val'], 2.0)
        b_val = _param(params, ['b_val', 'b'], 5.0)
        nu_val = _param(params, ['nu_induction', 'nu'], 0.02)
        delta_V_val = _param(params, ['delta_P', 'delta_V'], 0.0)
        dt_val = _param(params, ['dt'], 0.1)

        chi = _lysogeny_window(params, n)
        ell_val = lysogeny_hazard(
            _param(params, ['p_lys'], 0.0), w_val, int(sum(chi))
        )

        def const(val):
            return fem.Constant(domain, default_scalar_type(val))

        D_c, beta_c, beta_I_c, beta_L_c = (
            const(D_val), const(beta_val), const(beta_I_val), const(beta_L_val)
        )
        w_c, burst_c, nu_c = const(w_val), const(b_val), const(nu_val)
        delta_V_c, dt_c = const(delta_V_val), const(dt_val)
        ell = [chi_k * const(ell_val) for chi_k in chi]

        v = ufl.TestFunctions(W)
        uc = ufl.split(u)
        unc = ufl.split(u_n)
        dx = ufl.dx

        V, S = uc[0], uc[1]
        V_n, S_n = unc[0], unc[1]
        I_ = uc[2:2 + n]
        I_n = unc[2:2 + n]
        L, L_n = uc[2 + n], unc[2 + n]
        v_V, v_S, v_I, v_L = v[0], v[1], v[2:2 + n], v[2 + n]
        V_ads, removal = self._phage_cutoff(V, params, domain)

        eq_V = (
            ((V - V_n) / dt_c) * v_V * dx
            + D_c * ufl.inner(ufl.grad(V), ufl.grad(v_V)) * dx
            - (
                -beta_c * V_ads * S
                - beta_I_c * V_ads * sum(I_)
                - beta_L_c * V_ads * L
                + burst_c * w_c * I_[-1]
                - delta_V_c * V
            ) * v_V * dx
        )

        if removal is not None:
            eq_V += removal * v_V * dx

        eq_S = (
            ((S - S_n) / dt_c) * v_S * dx
            - (-beta_c * V_ads * S) * v_S * dx
        )

        eq_I = 0
        for k in range(n):
            if k == 0:
                src = beta_c * V_ads * S + nu_c * L
            else:
                src = w_c * I_[k - 1]
            eq_I += (
                ((I_[k] - I_n[k]) / dt_c) * v_I[k] * dx
                - (src - (w_c + ell[k]) * I_[k]) * v_I[k] * dx
            )

        eq_L = (
            ((L - L_n) / dt_c) * v_L * dx
            - (
                sum(ell[k] * I_[k] for k in range(n)) - nu_c * L
            ) * v_L * dx
        )

        return eq_V + eq_S + eq_I + eq_L

    @classmethod
    def dispersion_args(cls, params):
        return _lysogeny_dispersion_args(params)

    @classmethod
    def describe_theory(cls, params):
        return [_describe_lysogeny(params)]

    def profile_groups(self):
        return [
            ProfileGroup('S', (1,), r'$S$ (susceptible)', 'orange'),
            ProfileGroup('V', (0,), r'$V$ (phage)', 'purple', is_phage=True),
            ProfileGroup(
                'I', tuple(range(2, 2 + self.n_stages)),
                r'$I = \sum I_k$', 'teal',
            ),
            ProfileGroup(
                'L', (self.lysogen_subspace(),), r'$L$ (lysogens)', 'brown'
            ),
        ]


class ImmuneSignalingModel(BaseModel):
    """Delay chain with a death signal and an RNA-protein host response.

    First review, revision of Oct 24 2026, §§ 3.1-3.7 (eqs. 22-28). Every
    lysis, productive or resistant, releases M signal units; the signal F
    diffuses (D_s) and decays (delta_F F). Each spatial bin carries an
    intracellular RNA m and protein p, shared by every cell in the bin and
    not diffusing:

        dm/dt = alpha_m H(F) - delta_m m,   dp/dt = alpha_p m - delta_p p,
        H(F) = F^h / (K^h + F^h),           phi(p) = phi_max p / (K_p + p).

    phi(p) is the resistant proportion (resistant cells adsorb at beta but
    burst b_R < b). Switching between productive and resistant is fast and
    reversible, so every compartment still allowed to switch has resistant
    share phi(p) and is tracked by its total: the uninfected U (S = (1-phi)
    U, R = phi U) and the infected stages T_1 .. T_{n_R}. On leaving stage
    n_R an infection freezes, splitting (1 - phi) : phi into productive I_k
    and resistant J_k for k > n_R (eq. 27). n_R = 0 fixes the status at
    infection; n_R = n lets the bin convert infected cells until lysis,
    whose burst is then b_eff(phi) = b - (b - b_R) phi (eq. 28). Reversion
    is intrinsic (protein decay); there is no separate reversion rate.
    Optional phage decay delta_V and density cutoff as elsewhere.

    Subclasses add phage strategies through three hooks:
    ``_latency_rate``, ``_burst`` and ``_lysogeny_hazards``. Lysogens do
    not switch; an induced lysogen re-enters stage 1 and takes the local
    phi (it joins T_1 when n_R >= 1, or splits into I_1 / J_1 when
    n_R = 0; § 6.1).

    Component layout in the mixed function space:
        0: V (free phage)       1: F (signal)       2: U (uninfected)
        3: m (RNA)              4: p (protein)
        5 .. 5 + n_R - 1:       T_1 .. T_{n_R} (switching infected)
        then I_{n_R+1} .. I_n   (frozen productive)
        then J_{n_R+1} .. J_n   (frozen resistant)
        then L                  (lysogens; only when ``has_lysogens``)

    Parameter names (manuscript symbol = canonical key): D_s = D_signal,
    delta_F = gamma_signal, K = T_signal, M = signal_yield, h = hill_signal,
    alpha_m, delta_m, alpha_p, delta_p, phi_max, K_p, n_R (default n),
    b_R = b_abi. ``response_timescale`` divides alpha_m, delta_m, alpha_p
    and delta_p (stretches the priming curve, keeps p_inf and phi_inf).
    """

    name = 'immune_signaling'
    aliases = ('immune_signalling',)
    initial_condition = ImmuneSignalingInitialCondition
    reports_wave_speed = False
    trackers = (SignalFrontTracker,)
    has_signal = True
    legacy_profile_names = {
        'P': 'V', 'S': 'F', 'B_S': 'S', 'A': 'R', 'A_primed': 'R_m',
    }
    param_aliases = {
        **BaseModel.param_aliases,
        'D_sig': 'D_signal', 'gamma_sig': 'gamma_signal',
        'T_sig': 'T_signal',
        'yield_signal': 'signal_yield', 'b_sig': 'signal_yield',
        # Manuscript notation (F signal, RNA m, protein p, phi(p))
        'D_s': 'D_signal', 'gamma_s': 'gamma_signal',
        'delta_F': 'gamma_signal', 'K': 'T_signal',
        'M': 'signal_yield', 'h': 'hill_signal',
        'tau_R': 'response_timescale',
        'F_p': 'sigma_p', 'h_p': 'hill_p',
    }
    sweep_defaults = {
        **BaseModel.sweep_defaults,
        'D_signal': 1.0, 'gamma_signal': 0.1, 'T_signal': 0.1,
        'signal_yield': 1.0, 'hill_signal': 20.0,
        # Host response defaults: manuscript set A' (§ 8.1)
        'alpha_m': 1.0, 'delta_m': 1.0, 'alpha_p': 2.0, 'delta_p': 0.1,
        'phi_max': 1.0, 'K_p': 1.0, 'response_timescale': 1.0,
    }
    implied_by = (
        'D_signal', 'gamma_signal', 'T_signal', 'signal_yield',
        'alpha_m', 'delta_m', 'alpha_p', 'delta_p', 'phi_max', 'K_p', 'n_R',
    )

    def __init__(
        self, n_stages: int = 10, n_R: int = None, phi_max: float = 1.0,
        K_p: float = 1.0, k_abi: int = None,
    ):
        self.n_stages = n_stages
        self.n_R = n_stages if n_R is None else int(n_R)
        if not 0 <= self.n_R <= n_stages:
            raise ValueError(
                f"n_R must lie in 0 .. n_stages ({n_stages}); got {self.n_R}."
            )
        self.phi_max = float(phi_max)
        self.K_p = float(K_p)
        self.k_abi = n_stages

    # -- layout --------------------------------------------------------------

    _U, _M, _P = 2, 3, 4

    @property
    def _T_start(self):
        return 5

    @property
    def _I_start(self):
        return self._T_start + self.n_R

    @property
    def _J_start(self):
        return self._I_start + (self.n_stages - self.n_R)

    @property
    def _L_index(self):
        return self._J_start + (self.n_stages - self.n_R)

    @property
    def num_components(self) -> int:
        return self._L_index + (1 if self.has_lysogens else 0)

    def get_front_subspaces(self):
        # Uninfected cells U = S + R
        return [self._U]

    def get_signal_subspace(self):
        return 1

    def get_protein_subspace(self):
        return self._P

    def signal_threshold(self, params):
        """Host response threshold K (shared by weak form and metrics)."""
        return float(_param(params, ['T_signal', 'T_sig', 'K'], 0.1))

    def resistant_share(self, p):
        """phi(p) = phi_max p / (K_p + p) on numpy arrays."""
        p = np.maximum(np.asarray(p, dtype=float), 0.0)
        return self.phi_max * p / (self.K_p + p)

    # -- strategy hooks ----------------------------------------------------

    def _latency_rate(self, F, c):
        """Per-stage latent rate w (may depend on the local signal F)."""
        return c['w']

    def _burst(self, F, c):
        """Burst b of productive lysis (may depend on F)."""
        return c['b']

    def _lysogeny_hazards(self, F, c):
        """Per-stage lysogeny hazards sigma_k(F), or None (no lysogeny)."""
        return None

    def _extra_constants(self, const, params, base):
        """Subclass constants, merged into the constants dict ``c``."""
        return {}

    # -- weak form ---------------------------------------------------------

    def _rates(self, params):
        """Plain-float rate constants of the weak form, in creation order."""
        beta = _param(params, ['f_B_val', 'f_B', 'beta'], 1.0)
        host = HostResponse.from_params(params)
        return {
            'D': _param(params, ['D_val', 'D'], 1.0),
            'beta': beta,
            'beta_I': _param(params, ['f_I_val', 'f_I', 'beta_I'], beta),
            'beta_L': _param(params, ['f_L_val', 'f_L', 'beta_L'], beta),
            'w': _param(params, ['w_val'], 2.0),
            'b': _param(params, ['b_val', 'b'], 5.0),
            'b_R': _param(params, ['b_abi', 'b_S', 'b_R'], 0.0),
            'delta_V': _param(params, ['delta_P', 'delta_V'], 0.0),
            'D_s': _param(params, ['D_signal', 'D_sig', 'D_s'], 1.0),
            'delta_F': _param(
                params, ['gamma_signal', 'gamma_sig', 'gamma_s', 'delta_F'],
                0.1,
            ),
            'K': self.signal_threshold(params),
            'h': _param(params, ['hill_signal', 'h'], 20.0),
            'M': _param(
                params, ['signal_yield', 'yield_signal', 'b_sig', 'M'], 1.0
            ),
            'alpha_m': host.alpha_m, 'delta_m': host.delta_m,
            'alpha_p': host.alpha_p, 'delta_p': host.delta_p,
            'phi_max': self.phi_max, 'K_p': self.K_p,
            'dt': _param(params, ['dt'], 0.1),
        }

    def create_weak_formulation(self, domain, W, u, u_n, params):
        n, n_R = self.n_stages, self.n_R
        base = self._rates(params)

        def const(val):
            return fem.Constant(domain, default_scalar_type(val))

        c = {key: const(val) for key, val in base.items()}
        c.update(self._extra_constants(const, params, base))

        v = ufl.TestFunctions(W)
        uc = ufl.split(u)
        unc = ufl.split(u_n)
        dx = ufl.dx
        dt = c['dt']

        def ddt(i):
            return ((uc[i] - unc[i]) / dt) * v[i] * dx

        V, F, U = uc[0], uc[1], uc[self._U]
        m_rna, p_prot = uc[self._M], uc[self._P]
        V_ads, removal = self._phage_cutoff(V, params, domain)
        T_ = [uc[self._T_start + k] for k in range(n_R)]
        I_ = [uc[self._I_start + i] for i in range(n - n_R)]
        J_ = [uc[self._J_start + i] for i in range(n - n_R)]
        L = uc[self._L_index] if self.has_lysogens else None

        H = _hill(F, c['K'], c['h'])
        w = self._latency_rate(F, c)
        b = self._burst(F, c)
        sigma = self._lysogeny_hazards(F, c)
        if sigma is None:
            sigma = [0.0] * n
        p_pos = ufl.max_value(p_prot, 0.0)
        phi = c['phi_max'] * p_pos / (c['K_p'] + p_pos)

        def stage_total(k):
            """All infected cells in latent stage k + 1."""
            return T_[k] if k < n_R else I_[k - n_R] + J_[k - n_R]

        # Lysis from the last stage: switching (n_R = n) or frozen
        if n_R == n:
            lysis_prod = w * (1.0 - phi) * T_[-1]
            lysis_res = w * phi * T_[-1]
        else:
            lysis_prod = w * I_[-1]
            lysis_res = w * J_[-1]

        beta_V = c['beta'] * V_ads
        infected = sum(stage_total(k) for k in range(n))

        # Free phage V, eq. (23)
        rhs_V = (
            -beta_V * U
            - c['beta_I'] * V_ads * infected
            + b * lysis_prod + c['b_R'] * lysis_res
            - c['delta_V'] * V
        )
        if L is not None:
            rhs_V -= c['beta_L'] * V_ads * L
        if removal is not None:
            rhs_V -= removal
        form = (
            ddt(0)
            + c['D'] * ufl.inner(ufl.grad(V), ufl.grad(v[0])) * dx
            - rhs_V * v[0] * dx
        )

        # Signal F, eq. (24): every lysis, productive or resistant, gives M
        form += (
            ddt(1)
            + c['D_s'] * ufl.inner(ufl.grad(F), ufl.grad(v[1])) * dx
            - (c['M'] * (lysis_prod + lysis_res) - c['delta_F'] * F)
            * v[1] * dx
        )

        # Uninfected cells U, eq. (25)
        form += ddt(self._U) - (-beta_V * U) * v[self._U] * dx

        # RNA and protein (no diffusion), eq. (22)
        form += ddt(self._M) - (
            c['alpha_m'] * H - c['delta_m'] * m_rna
        ) * v[self._M] * dx
        form += ddt(self._P) - (
            c['alpha_p'] * m_rna - c['delta_p'] * p_prot
        ) * v[self._P] * dx

        # New infections (and induced lysogens) enter latent stage 1
        entry = beta_V * U
        if L is not None:
            entry = entry + c['gamma'] * L

        # Switching stages T_1 .. T_{n_R}, eq. (26)
        for k in range(n_R):
            idx = self._T_start + k
            src = entry if k == 0 else w * T_[k - 1]
            form += ddt(idx) - (
                src - (w + sigma[k]) * T_[k]
            ) * v[idx] * dx

        # Frozen stages n_R + 1 .. n, eq. (27): status set on entry
        for k in range(n_R, n):
            i = k - n_R
            if k == n_R:
                into = entry if n_R == 0 else w * T_[n_R - 1]
                src_I, src_J = (1.0 - phi) * into, phi * into
            else:
                src_I, src_J = w * I_[i - 1], w * J_[i - 1]
            for comp, src, idx in (
                (I_[i], src_I, self._I_start + i),
                (J_[i], src_J, self._J_start + i),
            ):
                form += ddt(idx) - (
                    src - (w + sigma[k]) * comp
                ) * v[idx] * dx

        # Lysogens L, eq. (68)
        if L is not None:
            l_idx = self._L_index
            form += ddt(l_idx) - (
                sum(sigma[k] * stage_total(k) for k in range(n))
                - c['gamma'] * L
            ) * v[l_idx] * dx

        return form

    @classmethod
    def from_params(cls, params):
        n_R = params.get('n_R')
        return cls(
            n_stages=int(params['n_stages']),
            n_R=None if n_R is None else int(n_R),
            phi_max=float(_param(params, ['phi_max'], 1.0)),
            K_p=float(_param(params, ['K_p'], 1.0)),
            k_abi=params.get('k_abi'),
        )

    @classmethod
    def describe_theory(cls, params):
        s = host_response.summary(params)
        n_R = params.get('n_R')
        lines = [
            "Host response (RNA-protein, § 3.2): "
            f"p_inf = {s['p_inf']:.4g}, phi_inf = {s['phi_inf']:.3f}, "
            f"t_1/2 = {s['t_half']:.3g}, t* = {s['t_star']:.3g} "
            f"(f_c = {s['f_c']:.3f}), t_rev = {s['t_rev']:.3g}",
            f"Ceiling (§ 3.6): b_eff_inf = {s['b_eff_inf']:.3f} -> "
            + ("stall possible" if s['stall_possible']
               else "no stall possible (ceiling slow wave)"),
            "Conversion window: n_R = "
            + (f"{n_R}" if n_R is not None else f"n = {params['n_stages']}"),
        ]
        return lines

    @classmethod
    def extinction_from_scales(cls, params, scales):
        """Phage dies out if the primed zone ahead of it is wide enough.

        The primed zone width is the signal decay length times the log
        of the signal released per burst over the priming threshold.
        """
        D_s = float(params.get('D_signal', params.get('D_sig', 1.0)))
        gamma_s = float(
            params.get('gamma_signal', params.get('gamma_sig', 0.1))
        )
        K = float(params.get('T_signal', params.get('T_sig', 0.1)))
        M = float(
            params.get(
                'signal_yield',
                params.get('yield_signal', params.get('b_sig', 1.0))
            )
        )

        sig_scale = max(1.1, (M * scales.b) / max(K, 1e-6))
        L_primed_eff = math.sqrt(D_s / max(gamma_s, 1e-4)) * math.log(
            sig_scale
        )

        p_end = scales.P_max * math.exp(-L_primed_eff / scales.decay_length)
        if L_primed_eff >= scales.L_primed_crit or p_end < scales.p_ext:
            return 1, p_end, scales.L_primed_crit / scales.c_eff
        return 0, p_end, float('nan')

    # -- outputs -------------------------------------------------------------

    def profile_groups(self):
        n, n_R = self.n_stages, self.n_R
        infected = tuple(range(self._T_start, self._J_start + n - n_R))
        groups = [
            # S and R are derived from U and phi(p) (derived_profiles)
            ProfileGroup('S', (), r'$S = (1-\phi)U$ (naive)', 'orange'),
            ProfileGroup('R', (), r'$R = \phi U$ (resistant)', 'red'),
            ProfileGroup('V', (0,), r'$V$ (phage)', 'purple', is_phage=True),
            ProfileGroup('F', (1,), r'$F$ (signal)', 'green'),
            ProfileGroup('I', infected, r'$\sum_k$ infected', 'teal'),
            ProfileGroup('J', (), r'resistant infected', 'gray'),
        ]
        if self.has_lysogens:
            groups.append(
                ProfileGroup('L', (self._L_index,), r'$L$ (lysogens)', 'brown')
            )
        # Cached only (not drawn): RNA, protein, switching and frozen
        # resistant infected, and the resistant share phi(p)
        groups += [
            ProfileGroup('m', (self._M,)),
            ProfileGroup('p', (self._P,)),
            ProfileGroup('phi', ()),
        ]
        if n_R > 0:
            groups.append(
                ProfileGroup('T', tuple(range(self._T_start, self._I_start)))
            )
        if n_R < n:
            frozen = tuple(range(self._J_start, self._J_start + n - n_R))
            groups.append(ProfileGroup('J_frozen', frozen))
        return groups

    def derived_profiles(self, snapshot):
        phi = self.resistant_share(snapshot['p'])
        U = snapshot['U']
        return {
            'phi': phi,
            'S': (1.0 - phi) * U,
            'R': phi * U,
            'J': snapshot.get('J_frozen', 0.0) + phi * snapshot.get('T', 0.0),
        }

    def speed_panel_series(self):
        return [SeriesSpec(
            'L_halo', r'$L_{halo}$', 'tab:green', axis='right',
            ylabel=r'Halo length $L_{halo}$',
        )]

    def kymograph_overlays(self, result):
        overlays = super().kymograph_overlays(result)
        edge = result.series.get('primed_edge_x', [])
        if len(edge) > 0:
            overlays.append(Overlay(
                'line', r'Primed edge ($\phi U \geq S_0/2$)',
                t=result.series['t'], x=edge,
                style=dict(color='black', linestyle='--'),
            ))
        return overlays


class ImmuneSignalingEvasionModel(ImmuneSignalingModel):
    """Signal model where the phage reads F to set latency and burst.

    Continuous sensing (manuscript § 4.5): the per-stage latent rate
    switches from w_fast = n / tau to w_slow = n / tau_slow as the local
    signal crosses the phage trigger F_p (Hill steepness h_p):

        w(F) = w_fast - (w_fast - w_slow) * H_p(F)

    Burst modes:
        'switch'   b(F) = b + (b_slow - b) * H_p(F)
        'frontier' b(F) = b_max * [1 - exp(-(tau(F) - tau_e) / tau_m)]_+
                   with tau(F) = n / w(F) and b_max = b_val
                   (maturation trade-off, eq. 33)

    Resistant lysis keeps the host-capped burst b_R. Deferral stretches
    every latent stage, including the switchable stages k <= n_R, so in
    signalled cells it also lengthens the conversion window (§ 4.5,
    § 5.5). Defaults (tau_slow = tau, b_slow = b, F_p = K) reduce the
    model to the non-adaptive ImmuneSignalingModel. F_p is the
    ``sigma_p`` parameter (alias ``F_p``) and h_p is ``hill_p`` (alias
    ``h_p``).
    """

    name = 'immune_signaling_evasion'
    aliases = ()
    implied_by = ()

    def _extra_constants(self, const, params, base):
        n = self.n_stages
        tau_fast = float(n) / float(base['w'])
        tau_slow = float(_param(params, ['tau_slow'], tau_fast))
        burst_mode = _param(params, ['burst_mode'], 'switch')
        if burst_mode not in ('switch', 'frontier'):
            raise ValueError(f"Unknown burst_mode '{burst_mode}'.")
        self._burst_mode = burst_mode
        return {
            'w_slow': const(float(n) / tau_slow),
            'b_slow': const(_param(params, ['b_slow'], base['b'])),
            'F_p': const(_param(params, ['sigma_p', 'F_p'], base['K'])),
            'h_p': const(_param(params, ['hill_p', 'h_p'], base['h'])),
            'tau_e': const(_param(params, ['tau_eclipse'], 0.2)),
            'tau_m': const(_param(params, ['tau_assembly'], 1.0)),
        }

    def _H_p(self, F, c):
        return _hill(F, c['F_p'], c['h_p'])

    def _latency_rate(self, F, c):
        return c['w'] - (c['w'] - c['w_slow']) * self._H_p(F, c)

    def _burst(self, F, c):
        if self._burst_mode == 'frontier':
            tau_F = self.n_stages / self._latency_rate(F, c)
            growth = 1.0 - ufl.exp(-(tau_F - c['tau_e']) / c['tau_m'])
            return c['b'] * ufl.max_value(growth, 0.0)
        return c['b'] + (c['b_slow'] - c['b']) * self._H_p(F, c)

    @classmethod
    def dispersion_args(cls, params):
        # No static Abi fraction; naive leading-edge speed c0
        return dict(dispersion_args(params), f_abi=0.0)

    @classmethod
    def extinction_from_scales(cls, params, scales):
        # The signal-front estimate assumes a non-adaptive phage
        return extinction.NAN_METRICS

    @classmethod
    def describe_theory(cls, params):
        return super().describe_theory(params) + [
            "(Naive leading-edge speed c0; the signal is not modelled.)"
        ]


class ImmuneSignalingLysogenyEvasionModel(ImmuneSignalingModel):
    """Signal model where a temperate phage reads F to lysogenize.

    First review (revision of Oct 24 2026), § 6.1, eqs. 68-70. The latent
    rate w stays fixed; a per-stage lysogeny hazard

        sigma_k(F) = chi_k [sigma_lo + (sigma_hi - sigma_lo) H_p(F)]

    diverts infected cells of every latent stage (switching T_k and
    frozen I_k + J_k alike) into lysogens L. Lysogens adsorb phage at
    beta_L (phage lost), do not switch, and induce at rate gamma into
    latent stage 1, where they take the bin's resistant share phi(p) at
    induction: they join T_1 when n_R >= 1, or split (1 - phi) : phi into
    I_1 / J_1 when n_R = 0. sigma_lo / sigma_hi are set from lysogeny
    probabilities ell = p_lys / ell_hi = p_lys_hi over the decision window
    chi (lys_window_start .. lys_window_end). Defaults p_lys_hi = p_lys
    (non-sensing temperate phage), F_p = K. Aliases: gamma = nu_induction,
    ell = p_lys, ell_hi = p_lys_hi.
    """

    name = 'immune_signaling_lysogeny_evasion'
    aliases = ()
    has_lysogens = True
    implied_by = ()
    param_aliases = {
        **ImmuneSignalingModel.param_aliases,
        'gamma': 'nu_induction', 'ell': 'p_lys', 'ell_hi': 'p_lys_hi',
    }

    def _extra_constants(self, const, params, base):
        chi = _lysogeny_window(params, self.n_stages)
        n_window = int(sum(chi))
        ell_lo = _param(params, ['p_lys', 'ell'], 0.0)
        ell_hi = _param(params, ['p_lys_hi', 'ell_hi'], ell_lo)
        self._chi = chi
        return {
            'gamma': const(_param(
                params, ['nu_induction', 'nu', 'gamma'], 0.02
            )),
            'sigma_lo': const(lysogeny_hazard(ell_lo, base['w'], n_window)),
            'sigma_hi': const(lysogeny_hazard(ell_hi, base['w'], n_window)),
            'F_p': const(_param(params, ['sigma_p', 'F_p'], base['K'])),
            'h_p': const(_param(params, ['hill_p', 'h_p'], base['h'])),
        }

    def _lysogeny_hazards(self, F, c):
        H_p = _hill(F, c['F_p'], c['h_p'])
        sigma = c['sigma_lo'] + (c['sigma_hi'] - c['sigma_lo']) * H_p
        return [chi_k * sigma for chi_k in self._chi]

    @classmethod
    def dispersion_args(cls, params):
        return _lysogeny_dispersion_args(params)

    @classmethod
    def extinction_from_scales(cls, params, scales):
        # The signal-front estimate assumes a non-adaptive phage
        return extinction.NAN_METRICS

    @classmethod
    def describe_theory(cls, params):
        return super().describe_theory(params) + [
            _describe_lysogeny(params),
            "(Naive leading-edge speed c0; the signal is not modelled.)",
        ]


# Every model the CLI, sweeps and history loader can construct
MODEL_CLASSES = (
    DelayChainModel,
    AbortiveInfectionModel,
    AbiHeterogeneousModel,
    LysogenyDelayChainModel,
    ImmuneSignalingModel,
    ImmuneSignalingEvasionModel,
    ImmuneSignalingLysogenyEvasionModel,
)
