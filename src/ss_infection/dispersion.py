import numpy as np
from scipy.optimize import minimize_scalar


def solve_dispersion_erlang(
    n, D, k, w, b, f_abi=0.0, grid_size=1000, lambda_max=5.0,
    hazards=None,
):
    """Calculates theoretical wave speed c* for Erlang delay-chain model.

    Solves the dispersion relation:
        (s - D*lambda^2 + k) * prod_k (s + w + ell_k)
            = k * b * (1 - alpha) * w^n
    where s = lambda * c, k = f_B * B0, alpha = f_abi is the
    fraction of the bacterial lawn carrying the Abi system, and ell_k are
    optional per-stage lysogeny hazards (review eq. 63; induction is
    neglected at the leading edge). With ell_k = 0 this is
    (s - D*lambda^2 + k)(s + w)^n = k * b * (1 - alpha) * w^n.

    Parameters:
        n (int): Number of latent stages (n >= 1).
        D (float): Phage diffusion coefficient.
        k (float): Effective phage capture rate (f_B * B0).
        w (float): Per-stage transition rate (w = n / tau).
        b (float): Burst size.
        f_abi (float): Fraction of lawn carrying Abi (alpha).
        grid_size (int): Number of grid points for lambda search.
        lambda_max (float): Maximum lambda to search.
        hazards (sequence of float): Per-stage lysogeny hazards ell_k
            (length n); None means no lysogeny.

    Returns:
        tuple: (c_star, lambda_star)
    """
    effective_b = (1.0 - f_abi) * b
    if hazards is None:
        hazards = np.zeros(n)
    hazards = np.asarray(hazards, dtype=float)
    # Leading-edge reproduction number including loss to lysogeny
    if effective_b * np.prod(w / (w + hazards)) <= 1.0:
        return 0.0, 0.0

    def residual(s, lam):
        stage = s + w + hazards
        prod = np.prod(stage)
        edge = s - D * lam**2 + k
        F = edge * prod - effective_b * k * (w**n)
        F_prime = prod * (1.0 + edge * np.sum(1.0 / stage))
        return F, F_prime

    lambdas = np.linspace(1e-4, lambda_max, grid_size)
    c_vals = np.zeros_like(lambdas)

    # Initial guess s0 at lambda -> 0+
    s = k * (effective_b - 1.0)

    for i, lam in enumerate(lambdas):
        for _ in range(100):
            F, F_prime = residual(s, lam)
            ds = F / F_prime
            s -= ds
            if abs(ds) < 1e-12:
                break
        c_vals[i] = s / lam

    min_idx = np.argmin(c_vals)
    lam_init = lambdas[min_idx]

    def obj(lam):
        if lam <= 0:
            return 1e9
        s_loc = c_vals[min_idx] * lam
        for _ in range(50):
            F, F_prime = residual(s_loc, lam)
            ds = F / F_prime
            s_loc -= ds
            if abs(ds) < 1e-12:
                break
        return s_loc / lam

    res = minimize_scalar(
        obj,
        bounds=(max(1e-4, lam_init * 0.5), lam_init * 1.5),
        method='bounded'
    )

    if res.success:
        lambda_star = float(res.x)
        c_star = float(res.fun)
    else:
        lambda_star = float(lam_init)
        c_star = float(c_vals[min_idx])

    return c_star, lambda_star


def solve_dispersion_fixed_delay(
    tau, D, k, b, f_abi=0.0, grid_size=1000, lambda_max=5.0
):
    """Calculates theoretical wave speed c* for fixed (deterministic) delay.

    Parameters:
        tau (float): Mean latent period.
        D (float): Phage diffusion coefficient.
        k (float): Effective phage capture rate (f_B * B0).
        b (float): Burst size.
        f_abi (float): Fraction of lawn carrying Abi.
        grid_size (int): Number of grid points for lambda search.
        lambda_max (float): Maximum lambda to search.

    Returns:
        tuple: (c_star, lambda_star)
    """
    effective_b = (1.0 - f_abi) * b
    if effective_b <= 1.0:
        return 0.0, 0.0

    lambdas = np.linspace(1e-4, lambda_max, grid_size)
    c_vals = np.zeros_like(lambdas)

    s = k * (effective_b - 1.0)

    for i, lam in enumerate(lambdas):
        for _ in range(100):
            F = s - D * lam**2 + k * (
                1.0 - effective_b * np.exp(-s * tau)
            )
            F_prime = 1.0 + k * effective_b * tau * np.exp(-s * tau)
            ds = F / F_prime
            s -= ds
            if abs(ds) < 1e-12:
                break
        c_vals[i] = s / lam

    min_idx = np.argmin(c_vals)
    lam_init = lambdas[min_idx]

    def obj(lam):
        if lam <= 0:
            return 1e9
        s_loc = c_vals[min_idx] * lam
        for _ in range(50):
            F = s_loc - D * lam**2 + k * (
                1.0 - effective_b * np.exp(-s_loc * tau)
            )
            F_prime = 1.0 + k * effective_b * tau * np.exp(-s_loc * tau)
            ds = F / F_prime
            s_loc -= ds
            if abs(ds) < 1e-12:
                break
        return s_loc / lam

    res = minimize_scalar(
        obj,
        bounds=(max(1e-4, lam_init * 0.5), lam_init * 1.5),
        method='bounded'
    )

    if res.success:
        lambda_star = float(res.x)
        c_star = float(res.fun)
    else:
        lambda_star = float(lam_init)
        c_star = float(c_vals[min_idx])

    return c_star, lambda_star


def dispersion_args(params):
    """Parse the solve_dispersion_erlang inputs shared by every model.

    Parameters:
        params (dict): Dictionary containing simulation parameters.

    Returns:
        dict: Keyword arguments n, D, k, w, b, f_abi (hazards unset).
    """
    n = int(params.get('n_stages', params.get('n', 10)))
    tau = float(params.get('tau', 5.0))
    D = float(params.get('D_val', params.get('D', 1.0)))
    f_B = float(params.get('f_B_val', params.get('f_B', 1.0)))
    B0 = float(params.get('B0_val', params.get('B0', 1.0)))
    b = float(
        params.get('b_val', params.get('b', params.get('burst_size', 5.0)))
    )
    f_abi = float(
        params.get(
            'f_abi', params.get('f_abi_val', params.get('alpha', 0.0))
        )
    )
    return dict(n=n, D=D, k=f_B * B0, w=n / tau, b=b, f_abi=f_abi)


def calculate_theoretical_speed(params):
    """Theoretical speed of the model named by ``params['model']``.

    Delegates to that model's ``theoretical_speed`` classmethod (see
    models.BaseModel).

    Parameters:
        params (dict): Dictionary containing simulation parameters.

    Returns:
        tuple: (c_star, lambda_star)
    """
    from .factory import get_model_class

    model_cls = get_model_class(params.get('model', 'delay_chain'))
    return model_cls.theoretical_speed(params)
