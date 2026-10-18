import abc

import numpy as np


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
            return 5.0 * np.exp(-x[0]**2 / 2.0)

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
            return 5.0 * np.exp(-x[0]**2 / 2.0)

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
    followed by a patch with f_abi abortive cells (L_naive < x <= L_total).

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

        def initial_V(x):
            return 5.0 * np.exp(-x[0]**2 / 2.0)

        def initial_S(x):
            res = np.empty(x.shape[1])
            is_naive = x[0] <= L_naive
            res[is_naive] = S0
            res[~is_naive] = (1.0 - f_abi) * S0
            return res

        def initial_R(x):
            res = np.empty(x.shape[1])
            is_naive = x[0] <= L_naive
            res[is_naive] = 0.0
            res[~is_naive] = f_abi * S0
            return res

        u.sub(0).interpolate(initial_V)
        u.sub(1).interpolate(initial_S)
        u.sub(2).interpolate(initial_R)

        # Sync ghost values for parallel MPI execution
        u.x.scatter_forward()


class ImmuneSignalingInitialCondition(BaseInitialCondition):
    """IC for the immune signaling model.

    Initialises S = S0 across the domain, V as a Gaussian bolus,
    and all signal, priming, and infected components to 0.

    Component layout expected:
        0: V, 1: F, 2: S, 3+: refractory chain, infected grid, abortive chain
    """

    def apply(self, W, u, params):
        S0 = params.get('B0_val', params.get('B0', 1.0))

        def initial_V(x):
            return 5.0 * np.exp(-x[0]**2 / 2.0)

        def initial_S(x):
            return np.full(x.shape[1], S0)

        u.sub(0).interpolate(initial_V)
        u.sub(2).interpolate(initial_S)

        # Sync ghost values for parallel MPI execution
        u.x.scatter_forward()
