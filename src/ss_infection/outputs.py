"""Output declarations a model makes about itself.

Models describe *what* to record and draw with these small records; the
simulation, history writer and plotting code consume them without knowing
which model produced them.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProfileGroup:
    """A spatial profile: the sum of some mixed-space components.

    Every group is cached in history files under ``name``. Groups with a
    ``label`` are also drawn in snapshot plots, in declaration order;
    ``is_phage`` marks the curve that may move to a secondary axis.
    """

    name: str
    indices: tuple
    label: str = None
    color: str = None
    is_phage: bool = False


@dataclass
class Overlay:
    """A line drawn on the kymograph (time on x, space on y).

    kind 'line':  ``t`` against ``x`` (NaN gaps break the line)
    kind 'hline': horizontal line at position ``y``
    """

    kind: str
    label: str
    t: object = None
    x: object = None
    y: float = None
    style: dict = field(default_factory=dict)


@dataclass
class SeriesSpec:
    """A per-step series (``result.series[key]``) drawn against time.

    ``axis`` is 'left' or 'right' (secondary y axis, labelled
    ``ylabel``). Line width and font sizes come from plotting.STYLE.
    """

    key: str
    label: str
    color: str
    axis: str = 'left'
    ylabel: str = None
    style: dict = field(default_factory=dict)
