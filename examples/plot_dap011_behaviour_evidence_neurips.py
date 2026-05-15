"""Make a NeurIPS-style DAP011 behaviour-only evidence comparison figure.

Compares:
  * GLM (shallow), zero-mean Gaussian walk
  * GLM (shallow) + REINFORCE
  * Deep + TE rule, side/zero input encoding

Run from the repo root:

    python examples/plot_dap011_behaviour_evidence_neurips.py
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/matplotlib-psytrax')
os.environ.setdefault('XDG_CACHE_HOME', '/private/tmp/psytrax-cache')
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
FITS_DIR = REPO_ROOT / 'fits'
FIG_DIR = REPO_ROOT / 'figures'

FIT_PATHS = {
    'GLM (shallow)': FITS_DIR / 'DAP011_behaviour_logistic_zero_mean_magnitude_fit.npy',
    'GLM (shallow) + REINFORCE': FITS_DIR / 'DAP011_behaviour_logistic_reinforce_side_fit.npy',
    'Deep + TE rule': FITS_DIR / 'DAP011_behaviour_tutor_executor_rule_side_fit.npy',
}


def load_rows():
    rows = []
    for label, path in FIT_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(
                f'Missing {path}. Run examples/compare_dap011_behaviour_learning_rules.py first.'
            )
        fit = np.load(path, allow_pickle=True).item()
        rows.append({
            'model': label,
            'fit_path': str(path.relative_to(REPO_ROOT)),
            'n_params': int(fit['params'].shape[0]),
            'n_trials': int(fit['n_trials']),
            'log_evidence': float(fit['log_evidence']),
        })
    baseline = rows[0]['log_evidence']
    for row in rows:
        row['delta_log_evidence_vs_glm'] = row['log_evidence'] - baseline
    return rows


def save_values(rows):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / 'dap011_behaviour_evidence_neurips_values.csv'
    with open(out, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out


def plot(rows):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    labels = ['GLM (shallow)', 'GLM (shallow) + R', 'Deep + TE rule']
    deltas = np.array([row['delta_log_evidence_vs_glm'] for row in rows])
    log_evidence = np.array([row['log_evidence'] for row in rows])
    y = np.arange(len(rows))

    colors = ['#7A7A7A', '#4C78A8', '#54A24B']

    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fig, ax = plt.subplots(figsize=(3.45, 2.0))
    fig.subplots_adjust(left=0.34, right=0.98, bottom=0.25, top=0.96)
    ax.barh(y, deltas, color=colors, height=0.52, edgecolor='none')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(r'$\Delta$ log evidence vs. GLM (shallow) (nats)')
    ax.set_xlim(-25, max(deltas) * 1.06)
    ax.grid(axis='x', color='#D9D9D9', linewidth=0.6)
    ax.set_axisbelow(True)

    for yi, delta in zip(y, deltas):
        if delta > 50:
            x = delta - max(deltas) * 0.025
            ha = 'right'
            text_color = 'white'
        elif delta < 1:
            x = 8.0
            ha = 'left'
            text_color = '#222222'
        else:
            x = delta + max(deltas) * 0.018
            ha = 'left'
            text_color = '#222222'
        ax.text(
            x,
            yi,
            f'{delta:+.2f}',
            va='center',
            ha=ha,
            fontsize=7.5,
            color=text_color,
        )

    ax.text(
        0.98,
        0.96,
        f'GLM log ev. = {log_evidence[0]:.1f}',
        transform=ax.transAxes,
        va='top',
        ha='right',
        fontsize=6.5,
        color='#555555',
    )

    stem = FIG_DIR / 'dap011_behaviour_evidence_neurips'
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.png'), dpi=600)
    plt.close(fig)
    return stem


def main():
    rows = load_rows()
    values_path = save_values(rows)
    stem = plot(rows)
    print(f'Wrote {values_path.relative_to(REPO_ROOT)}')
    print(f'Wrote {stem.relative_to(REPO_ROOT)}.pdf')
    print(f'Wrote {stem.relative_to(REPO_ROOT)}.svg')
    print(f'Wrote {stem.relative_to(REPO_ROOT)}.png')
    for row in rows:
        print(
            f"{row['model']}: log_evidence={row['log_evidence']:.3f}, "
            f"delta={row['delta_log_evidence_vs_glm']:+.3f}"
        )


if __name__ == '__main__':
    main()
