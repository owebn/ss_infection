"""Model registry shared by the CLI, sweeps and history loader.

Everything here is generated from ``models.MODEL_CLASSES``; to add a model,
define its class (see ``models.BaseModel``) and list it there.
"""

from .models import BASE_OUTPUT_COLUMNS, MODEL_CLASSES  # noqa: F401

# Every accepted model name (canonical names and aliases) -> class
MODEL_REGISTRY = {
    name: cls
    for cls in MODEL_CLASSES
    for name in (cls.name,) + tuple(cls.aliases)
}

# Canonical names, in declaration order (CLI choices)
MODEL_NAMES = tuple(cls.name for cls in MODEL_CLASSES)

SIGNALING_MODELS = tuple(
    name for name, cls in MODEL_REGISTRY.items()
    if getattr(cls, 'has_signal', False)
)

LYSOGENY_MODELS = tuple(
    name for name, cls in MODEL_REGISTRY.items() if cls.has_lysogens
)

# Sweep parameter aliases of every model (alias -> canonical name)
PARAM_ALIASES = {}
for _cls in MODEL_CLASSES:
    for _alias, _canonical in _cls.param_aliases.items():
        PARAM_ALIASES.setdefault(_alias, _canonical)


def get_model_class(model_name):
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {', '.join(MODEL_NAMES)}"
        ) from None


def infer_model(param_names):
    """Model a sweep selects through its parameter names, or None.

    Checks models' ``implied_by`` in reverse declaration order, so that
    more specific models (declared later) win.
    """
    param_names = set(param_names)
    for cls in reversed(MODEL_CLASSES):
        if param_names.intersection(cls.implied_by):
            return cls.name
    return None


def output_columns(model_name, theoretical=False):
    """Result columns a run of ``model_name`` reports, in CSV order."""
    return get_model_class(model_name).output_columns(theoretical)


def build_model(params):
    """Return (model, initial_condition) for ``params['model']``."""
    cls = get_model_class(params.get('model', 'delay_chain'))
    return cls.from_params(params), cls.initial_condition()
