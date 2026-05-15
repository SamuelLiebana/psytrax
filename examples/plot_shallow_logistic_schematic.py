"""Draw a compact schematic of the shallow logistic choice model."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/matplotlib-psytrax')
os.environ.setdefault('XDG_CACHE_HOME', '/private/tmp/psytrax-cache')
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / 'figures'


def _node(ax, xy, label, *, radius=0.09, facecolor='white',
          edgecolor='#222222', fontsize=9, linewidth=1.4):
    circ = Circle(xy, radius, facecolor=facecolor, edgecolor=edgecolor,
                  linewidth=linewidth)
    ax.add_patch(circ)
    ax.text(*xy, label, ha='center', va='center', fontsize=fontsize)


def _arrow(ax, start, end, *, label=None, label_xy=None, color='#222222',
           linewidth=1.25):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle='-|>',
        mutation_scale=9,
        linewidth=linewidth,
        color=color,
        shrinkA=8,
        shrinkB=8,
    )
    ax.add_patch(arrow)
    if label is not None:
        ax.text(*label_xy, label, ha='center', va='center', fontsize=8,
                color=color)


def plot():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.size': 8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fig, ax = plt.subplots(figsize=(3.45, 1.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    input_color = '#D9E8F5'
    readout_color = '#E8E8E8'
    output_color = '#E2F0DE'

    c_xy = (0.13, 0.66)
    bias_xy = (0.13, 0.34)
    sum_xy = (0.46, 0.50)
    sig_xy = (0.68, 0.50)
    out_xy = (0.88, 0.50)

    ax.text(0.02, 0.92, 'Shallow logistic', ha='left', va='top',
            fontsize=9, fontweight='bold')

    _node(ax, c_xy, r'$c_t$', facecolor=input_color)
    _node(ax, bias_xy, r'$1$', facecolor=input_color)
    _node(ax, sum_xy, r'$\sum$', facecolor=readout_color)
    _node(ax, sig_xy, r'$\sigma$', facecolor=readout_color)
    _node(ax, out_xy, r'$P(R_t)$', radius=0.095, facecolor=output_color)

    _arrow(ax, c_xy, sum_xy, label=r'$w_t$', label_xy=(0.29, 0.65))
    _arrow(ax, bias_xy, sum_xy, label=r'$b_t$', label_xy=(0.29, 0.35))
    _arrow(ax, sum_xy, sig_xy)
    _arrow(ax, sig_xy, out_xy)

    ax.text(0.58, 0.18, r'$P(R_t)=\sigma(w_t c_t+b_t)$',
            ha='center', va='center', fontsize=8)

    stem = FIG_DIR / 'shallow_logistic_schematic'
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(stem.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(stem.with_suffix('.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)
    return stem


def main():
    stem = plot()
    print(f'Wrote {stem.relative_to(REPO_ROOT)}.pdf')
    print(f'Wrote {stem.relative_to(REPO_ROOT)}.svg')
    print(f'Wrote {stem.relative_to(REPO_ROOT)}.png')


if __name__ == '__main__':
    main()
