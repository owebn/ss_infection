"""Figures for a SimulationResult, with one shared style.

Every figure takes a SimulationResult (live or loaded with
SimulationResult.from_history) and gets its model-specific content from
the model's declarations (profile_groups, kymograph_overlays,
speed_panel_series), so new models need no changes here.

Building blocks (reuse these for new figures):
    STYLE               line width, fonts, figure sizes, shading, dpi
    new_axes()          white figure + axes
    style_axes()        labels, title and tick sizes
    draw_line()         a line with the shared line width
    draw_series()       a SeriesSpec from result.series
    shade_intervals()   semi-transparent spans (e.g. stalls)
    merged_legend()     one legend for an axes and its twin
    finish()            layout, save, show/close

Figures (FIGURES maps file stem -> function):
    kymograph   uninfected density in space-time, model overlays
    speed       smoothed tip speed, c_ref, stalls shaded; model series
                (e.g. L_halo) on a secondary axis
    phage       total free phage over time (log scale), stalls shaded
    snapshot    labelled profiles at the final snapshot

save_run_outputs(result, out_dir) writes every figure plus metrics.json
and params.json.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from . import regimes

STYLE = {
    'linewidth': 2.0,
    'figsize': (10, 5),
    'kymograph_figsize': (10, 6),
    'title_size': 13,
    'label_size': 12,
    'tick_size': 10,
    'legend_size': 10,
    'legend_alpha': 0.9,
    'dpi': 300,
    'stall': dict(color='red', alpha=0.2, linewidth=0),
    'reference': dict(color='gray', linestyle=':'),
}

KYMOGRAPH_CMAP = LinearSegmentedColormap.from_list(
    'blue_to_red', ['#0000ff', '#ffffff', '#ff0000']
)


# -- building blocks ---------------------------------------------------------

def new_axes(figsize=None):
    fig, ax = plt.subplots(
        figsize=figsize or STYLE['figsize'], facecolor='white'
    )
    ax.set_facecolor('white')
    return fig, ax


def style_axes(ax, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=STYLE['label_size'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=STYLE['label_size'])
    if title:
        ax.set_title(title, fontsize=STYLE['title_size'])
    ax.tick_params(labelsize=STYLE['tick_size'])


def draw_line(ax, x, y, label=None, color=None, **style):
    """Plot a line using the shared line width unless overridden."""
    style.setdefault('linewidth', STYLE['linewidth'])
    return ax.plot(x, y, label=label, color=color, **style)[0]


def draw_series(ax, result, spec):
    """Draw outputs.SeriesSpec ``spec`` against result.series['t']."""
    values = result.series.get(spec.key)
    if values is None or len(values) != len(result.series['t']):
        return None
    return draw_line(
        ax, result.series['t'], values, label=spec.label, color=spec.color,
        **spec.style,
    )


def shade_intervals(ax, intervals, label='Stall'):
    """Semi-transparent vertical spans; one legend entry for all."""
    for i, (t0, t1) in enumerate(intervals):
        ax.axvspan(t0, t1, label=label if i == 0 else None, **STYLE['stall'])


def merged_legend(*axes, loc='upper right'):
    handles, labels = [], []
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        handles += h
        labels += lab
    if handles:
        axes[0].legend(
            handles, labels, loc=loc, fontsize=STYLE['legend_size'],
            framealpha=STYLE['legend_alpha'], facecolor='white',
        )


def finish(fig, save_path=None, show=True):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=STYLE['dpi'], facecolor='white')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _time_axes_limits(ax, result):
    t = result.series.get('t', result.t_snap)
    if len(t):
        ax.set_xlim(0, float(np.max(t)))


# -- figures -----------------------------------------------------------------

def plot_kymograph(result, save_path=None, show=True):
    """Space-time heatmap of uninfected density U (% of S0).

    Lines come from result.model.kymograph_overlays(result) (e.g. the
    c* fit, patch boundaries, the primed edge); the title from
    result.model.kymograph_title(result).
    """
    fig, ax = new_axes(STYLE['kymograph_figsize'])

    T_grid, X_grid = np.meshgrid(result.t_snap, result.x)
    mesh_plot = ax.pcolormesh(
        T_grid, X_grid, result.kymograph_field(),
        cmap=KYMOGRAPH_CMAP, vmin=0.0, vmax=100.0, shading='auto'
    )
    cbar = fig.colorbar(mesh_plot, ax=ax)
    cbar.set_label(
        'Uninfected Bacterial Density (%)', fontsize=STYLE['label_size']
    )

    for overlay in result.model.kymograph_overlays(result):
        style = dict(overlay.style)
        style.setdefault('linewidth', STYLE['linewidth'])
        if overlay.kind == 'hline':
            ax.axhline(y=overlay.y, label=overlay.label, **style)
        else:
            ax.plot(overlay.t, overlay.x, label=overlay.label, **style)

    style_axes(
        ax, 'Time ($t$)', 'Space ($x$)',
        result.model.kymograph_title(result),
    )
    ax.set_xlim(0, max(result.t_snap))
    ax.set_ylim(0, result.params.get('L', max(result.x)))
    merged_legend(ax)
    return finish(fig, save_path, show)


def plot_speed(result, save_path=None, show=True, analysis=None):
    """Smoothed tip speed over time with stalls shaded.

    Draws the reference speed c_ref (fully susceptible lawn) and any
    result.model.speed_panel_series() (left or secondary right axis,
    e.g. L_halo for signal models).
    """
    analysis = analysis or regimes.analyze(result)
    t = result.series['t']
    fig, ax = new_axes()

    shade_intervals(ax, analysis.stalls)
    draw_line(ax, t, analysis.speed, label='Tip speed', color='tab:blue')
    c_ref = analysis.metrics['c_ref']
    if np.isfinite(c_ref):
        ax.axhline(
            c_ref, label=f'$c_{{ref}} = {c_ref:.3g}$',
            linewidth=STYLE['linewidth'], **STYLE['reference'],
        )

    ax_right = None
    for spec in result.model.speed_panel_series():
        target = ax
        if spec.axis == 'right':
            if ax_right is None:
                ax_right = ax.twinx()
            target = ax_right
            if spec.ylabel:
                ax_right.set_ylabel(
                    spec.ylabel, fontsize=STYLE['label_size'],
                    color=spec.color,
                )
                ax_right.tick_params(
                    axis='y', labelcolor=spec.color,
                    labelsize=STYLE['tick_size'],
                )
        draw_series(target, result, spec)

    style_axes(
        ax, 'Time ($t$)', 'Tip speed',
        f'Front speed (regime: {analysis.regime})',
    )
    _time_axes_limits(ax, result)
    merged_legend(*[a for a in (ax, ax_right) if a is not None])
    return finish(fig, save_path, show)


def plot_phage_totals(result, save_path=None, show=True, analysis=None):
    """Total free phage (integral of V over space) over time, log scale."""
    analysis = analysis or regimes.analyze(result)
    t = result.series['t']
    total = np.asarray(result.series['p_total'], dtype=float)
    fig, ax = new_axes()

    shade_intervals(ax, analysis.stalls)
    positive = np.where(total > 0, total, np.nan)
    draw_line(ax, t, positive, label=r'$\int V\,dx$', color='purple')
    if np.any(np.isfinite(positive)):
        ax.set_yscale('log')

    style_axes(ax, 'Time ($t$)', 'Total free phage', 'Free phage over time')
    _time_axes_limits(ax, result)
    merged_legend(ax)
    return finish(fig, save_path, show)


def plot_snapshot(result, index=-1, save_path=None, show=True):
    """Labelled profiles of result.model.profile_groups() at one snapshot.

    The phage group moves to a secondary axis when its scale differs from
    the first group's by more than 5x.
    """
    groups = [g for g in result.model.profile_groups()
              if g.label and g.name in result.profiles]
    x = result.x
    values = [result.profiles[g.name][index] for g in groups]

    fig, ax1 = new_axes()
    phage_idx = next((i for i, g in enumerate(groups) if g.is_phage), None)

    ax2 = None
    if phage_idx is not None:
        p_max = float(np.max(values[phage_idx])) if len(x) else 0.0
        b_max = float(np.max(values[0])) if len(x) else 1.0
        if p_max > 5.0 * max(b_max, 1.0) or (
            p_max > 0 and b_max / max(p_max, 1e-9) > 5.0
        ):
            ax2 = ax1.twinx()

    for i, (group, vals) in enumerate(zip(groups, values)):
        if ax2 is not None and i == phage_idx:
            draw_line(ax2, x, vals, label=group.label + ' (right axis)',
                      color=group.color)
            ax2.set_ylabel('Phage Concentration',
                           fontsize=STYLE['label_size'], color=group.color)
            ax2.tick_params(axis='y', labelcolor=group.color,
                            labelsize=STYLE['tick_size'])
        else:
            draw_line(ax1, x, vals, label=group.label, color=group.color)

    style_axes(
        ax1, 'Space ($x$)', 'Density / Concentration',
        'Spatial Profile Snapshot ($t = {:.1f}$)'.format(
            float(result.t_snap[index])
        ),
    )
    ax1.set_xlim(0, max(x))
    merged_legend(*[a for a in (ax1, ax2) if a is not None])
    return finish(fig, save_path, show)


FIGURES = {
    'kymograph': plot_kymograph,
    'speed': plot_speed,
    'phage': plot_phage_totals,
    'snapshot': plot_snapshot,
}


def save_run_outputs(result, out_dir, figures=None, show=False):
    """Write every figure plus metrics.json and params.json to out_dir.

    ``figures`` maps file stem -> plotting function (default FIGURES).
    Returns the list of files written.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for stem, plot in (figures or FIGURES).items():
        path = os.path.join(out_dir, f'{stem}.png')
        plot(result, save_path=path, show=show)
        written.append(path)

    analysis = regimes.analyze(result)
    summary = {
        **result.metrics,
        'stalls': [list(s) for s in analysis.stalls],
        'success': bool(result.success),
    }
    for name, payload in (('metrics', summary), ('params', result.params)):
        path = os.path.join(out_dir, f'{name}.json')
        with open(path, 'w') as f:
            json.dump(_to_json(payload), f, indent=1)
        written.append(path)
    return written


def _to_json(value):
    """JSON-safe copy: numpy scalars to Python, NaN/inf to null."""
    if isinstance(value, dict):
        return {str(k): _to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)
