import abc

import numpy as np


def bolus_amplitude(params):
    """Peak of the initial phage bolus A exp(-x^2 / 2) (default 5)."""
    val = params.get('bolus_amplitude')
    return 5.0 if val is None else float(val)


class BaseInitialCondition(abc.ABC):
    """Abstract base class for applying initial conditions."""

    @abc.abstractmethod
    def apply(self, W, u, params):
        """Apply initial conditions to u (W: mixed function space)."""
        pass


class BolusInitialCondition(BaseInitialCondition):
    """Free phage V as a Gaussian bolus on an untouched lawn S = S0."""

    def apply(self, W, u, params):
        S0 = params.get('B0_val', 1.0)

        def initial_V(x):
            return bolus_amplitude(params) * np.exp(-x[0]**2 / 2.0)

        def initial_S(x):
            return np.full(x.shape[1], S0)

        u.sub(0).interpolate(initial_V)
        u.sub(1).interpolate(initial_S)

        # Sync ghost values for parallel MPI execution
        u.x.scatter_forward()


class AbiBolusInitialCondition(BaseInitialCondition):
    """IC for the abortive infection model.

    Splits the lawn into susceptible (S) and abortive (R)
    populations based on the f_abi fraction.  Free phage
    is initialised as a localised Gaussian bolus.

    Component layout expected:
        0: V, 1: S, 2: R, 3+: infected stages
    """

    def apply(self, W, u, params):
        S0 = params.get('B0_val', params.get('B0', 1.0))
        f_abi = float(
            params.get(
                'f_abi', params.get('f_abi_val', params.get('alpha', 0.5))
            )
        )

        def initial_V(x):
            return bolus_amplitude(params) * np.exp(-x[0]**2 / 2.0)

        def initial_S(x):
            return np.full(x.shape[1], (1.0 - f_abi) * S0)

        def initial_R(x):
            return np.full(x.shape[1], f_abi * S0)

        u.sub(0).interpolate(initial_V)
        u.sub(1).interpolate(initial_S)
        u.sub(2).interpolate(initial_R)

        # Sync ghost values for parallel MPI execution
        u.x.scatter_forward()


class AbiHeterogeneousInitialCondition(BaseInitialCondition):
    """IC for the spatially heterogeneous abortive infection model.

    Domain consists of a naive non-abortive region (0 <= x <= L_naive)
    followed by a patch with f_abi abortive cells of length L_primed,
    then naive lawn again for the last L_after (default 0).

    Component layout expected:
        0: V, 1: S, 2: R, 3+: infected stages
    """

    def apply(self, W, u, params):
        S0 = params.get('B0_val', params.get('B0', 1.0))
        f_abi = float(
            params.get(
                'f_abi', params.get('f_abi_val', params.get('alpha', 0.5))
            )
        )
        L_naive = float(
            params.get('L_naive', params.get('l_naive', 50.0))
        )
        L_after = float(params.get('L_after') or 0.0)
        patch_end = float(params['L']) - L_after if L_after > 0 else np.inf

        def in_patch(x):
            return (x[0] > L_naive) & (x[0] <= patch_end)

        def initial_V(x):
            return bolus_amplitude(params) * np.exp(-x[0]**2 / 2.0)

        def initial_S(x):
            return np.where(in_patch(x), (1.0 - f_abi) * S0, S0)

        def initial_R(x):
            return np.where(in_patch(x), f_abi * S0, 0.0)

        u.sub(0).interpolate(initial_V)
        u.sub(1).interpolate(initial_S)
        u.sub(2).interpolate(initial_R)

        # Sync ghost values for parallel MPI execution
        u.x.scatter_forward()


class ImmuneSignalingInitialCondition(BaseInitialCondition):
    """IC for the immune signaling model.

    Initialises the uninfected lawn U = S0 across the domain, V as a
    Gaussian bolus, and the signal, RNA, protein and infected components
    to 0. Optional pre-primed patch: ``primed_patch_p`` = p0 sets the
    protein to p0 on [``primed_patch_start``, ``primed_patch_end``]
    (RNA stays 0, so the patch decays at delta_p without signal).

    Component layout expected:
        0: V, 1: F, 2: U, 3: m (RNA), 4: p (protein), 5+: infected stages
    """

    protein_index = 4

    def apply(self, W, u, params):
        S0 = params.get('B0_val', params.get('B0', 1.0))

        def initial_V(x):
            return bolus_amplitude(params) * np.exp(-x[0]**2 / 2.0)

        def initial_S(x):
            return np.full(x.shape[1], S0)

        u.sub(0).interpolate(initial_V)
        u.sub(2).interpolate(initial_S)

        # Optional pre-primed patch: protein p = p0 on [start, end]
        p0 = params.get('primed_patch_p')
        if p0 is not None and float(p0) > 0:
            x0 = float(params.get('primed_patch_start') or 0.0)
            x1 = float(params.get('primed_patch_end') or 0.0)

            def initial_p(x):
                inside = (x[0] >= x0) & (x[0] <= x1)
                return np.where(inside, float(p0), 0.0)

            u.sub(self.protein_index).interpolate(initial_p)

        # Sync ghost values for parallel MPI execution
        u.x.scatter_forward()


class ImmuneSignalingSwitchingInitialCondition(
    ImmuneSignalingInitialCondition
):
    """IC for the finite-switching signalling model: the lawn starts
    naive (S = S0, R = 0), protein at component 5.

    Component layout expected:
        0: V, 1: F, 2: S, 3: R, 4: m, 5: p, 6+: infected stages
    """

    protein_index = 5
