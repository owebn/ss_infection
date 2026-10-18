import math
from typing import NamedTuple

NAN_METRICS = (float('nan'), float('nan'), float('nan'))


class EdgeScales(NamedTuple):
    """Leading-edge quantities shared by the extinction estimates."""

    b: float
    p_ext: float
    P_max: float
    decay_length: float
    L_primed_crit: float
    c_eff: float


def edge_scales(model_cls, params):
    """Leading-edge scales for ``model_cls``, or final metrics if trivial.

    Analytical Derivations:
        P_max = B0 * (b - 1)
        decay_length = sqrt(D / gamma)  where gamma = f_B * B0
        L_primed_crit = sqrt(D / gamma) * ln( B0 * (b - 1) / p_ext )

    Returns:
        EdgeScales, or an (extinction, p_end, t_extinction) tuple when the
        outcome does not depend on the model (no edge, or P_max <= p_ext).
    """
    D = float(params.get('D_val', params.get('D', 1.0)))
    f_B = float(params.get('f_B_val', params.get('f_B', 1.0)))
    B0 = float(params.get('B0_val', params.get('B0', 1.0)))
    b = float(params.get('b_val', params.get('b', 5.0)))
    p_ext_def = params.get('P_extinction', 1e-6)
    p_ext = float(
        params.get(
            'p_ext', params.get('p_extinction', p_ext_def)
        )
    )

    gamma = f_B * B0
    if gamma <= 0.0 or b <= 1.0 or p_ext <= 0.0:
        return NAN_METRICS

    P_max = B0 * (b - 1.0)
    if P_max <= p_ext:
        return 1, P_max, 0.0

    decay_length = math.sqrt(D / gamma)
    L_primed_crit = decay_length * math.log(P_max / p_ext)

    c_star, _ = model_cls.theoretical_speed(params)
    c_eff = max(c_star, 1e-3)

    return EdgeScales(
        b=b, p_ext=p_ext, P_max=P_max, decay_length=decay_length,
        L_primed_crit=L_primed_crit, c_eff=c_eff,
    )


def calculate_extinction_metrics(params):
    """Theoretical (extinction, p_end, t_extinction) for params['model'].

    Delegates to that model's ``theoretical_extinction`` classmethod (see
    models.BaseModel); models without an estimate return NaNs.

    Returns:
        tuple: (extinction, p_end, t_extinction)
    """
    from .factory import get_model_class

    model_cls = get_model_class(params.get('model', 'delay_chain'))
    return model_cls.theoretical_extinction(params)
