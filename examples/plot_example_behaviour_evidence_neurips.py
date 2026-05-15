"""Plot average example-animal behaviour evidence comparison.

Reads the long-form CSV written by
``examples/compare_dap011_behaviour_learning_rules.py --all-example-subjects``
and plots mean ± SEM delta log evidence relative to the shallow GLM within
each animal.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/matplotlib-psytrax')
os.environ.setdefault('XDG_CACHE_HOME', '/private/tmp/psytrax-cache')
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
FITS_DIR = REPO_ROOT / 'fits'
FIG_DIR = REPO_ROOT / 'figures'

MODEL_ORDER = [
    'glm_shallow_zero_mean',
    'glm_shallow_reinforce',
    'deep_zero_mean',
    'deep_te_rule',
]

MODEL_LABELS = {
    'glm_shallow_zero_mean': 'GLM (shallow)',
    'glm_shallow_reinforce': 'GLM (shallow) + R',
    'deep_zero_mean': 'Deep, 0-mean',
    'deep_te_rule': 'Deep + TE rule',
}

MODEL_COLORS = {
    'glm_shallow_zero_mean': '#7A7A7A',
    'glm_shallow_reinforce': '#4C78A8',
    'deep_zero_mean': '#8F63A9',
    'deep_te_rule': '#54A24B',
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--summary-csv',
        type=Path,
        default=FITS_DIR / 'example_behaviour_learning_rule_comparison_side.csv',
        help='Long-form evidence summary CSV from the comparison script.',
    )
    parser.add_argument(
        '--output-stem',
        type=Path,
        default=FIG_DIR / 'example_behaviour_evidence_neurips',
        help='Output path without extension.',
    )
    parser.add_argument(
        '--per-trial',
        action='store_true',
        help='Plot delta log evidence per trial, which is easier to average across animals.',
    )
    return parser.parse_args()


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT)
    except ValueError:
        return resolved


def load_subject_deltas(summary_csv: Path, per_trial: bool = False) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    required = {'subject', 'model_id', 'log_evidence'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{summary_csv} is missing required columns: {sorted(missing)}')

    pivot = df.pivot_table(
        index='subject',
        columns='model_id',
        values='log_evidence',
        aggfunc='first',
    )
    missing_models = [m for m in MODEL_ORDER if m not in pivot.columns]
    if missing_models:
        raise ValueError(
            f'{summary_csv} is missing model(s): '
            f'{", ".join(MODEL_LABELS[m] for m in missing_models)}'
        )
    pivot = pivot[MODEL_ORDER].dropna()
    if pivot.empty:
        raise ValueError(f'{summary_csv} does not contain any complete subject rows')

    baseline = pivot['glm_shallow_zero_mean']
    deltas = pivot.subtract(baseline, axis=0)
    if per_trial:
        n_trials = df.pivot_table(
            index='subject',
            columns='model_id',
            values='n_trials',
            aggfunc='first',
        )['glm_shallow_zero_mean'].reindex(deltas.index)
        deltas = deltas.divide(n_trials, axis=0)
    deltas = deltas.reset_index().melt(
        id_vars='subject',
        var_name='model_id',
        value_name='delta_log_evidence_vs_glm',
    )
    raw = pivot.reset_index().melt(
        id_vars='subject',
        var_name='model_id',
        value_name='log_evidence',
    )
    out = deltas.merge(raw, on=['subject', 'model_id'])
    out['model'] = out['model_id'].map(MODEL_LABELS)
    return out


def summarize(deltas: pd.DataFrame) -> list[dict]:
    rows = []
    for model_id in MODEL_ORDER:
        vals = deltas.loc[
            deltas['model_id'] == model_id,
            'delta_log_evidence_vs_glm',
        ].to_numpy(float)
        raw = deltas.loc[deltas['model_id'] == model_id, 'log_evidence'].to_numpy(float)
        n = vals.size
        sem = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        raw_sem = float(np.std(raw, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        rows.append({
            'model_id': model_id,
            'model': MODEL_LABELS[model_id],
            'n_subjects': int(n),
            'mean_log_evidence': float(np.mean(raw)),
            'sem_log_evidence': raw_sem,
            'mean_delta_log_evidence_vs_glm': float(np.mean(vals)),
            'sem_delta_log_evidence_vs_glm': sem,
        })
    return rows


def save_values(deltas: pd.DataFrame, rows: list[dict], output_stem: Path) -> tuple[Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    values_path = output_stem.with_name(output_stem.name + '_values.csv')
    subject_path = output_stem.with_name(output_stem.name + '_subject_values.csv')
    with open(values_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    deltas.to_csv(subject_path, index=False)
    return values_path, subject_path


def plot(deltas: pd.DataFrame, rows: list[dict], output_stem: Path,
         per_trial: bool = False) -> None:
    means = np.array([row['mean_delta_log_evidence_vs_glm'] for row in rows])
    sems = np.array([row['sem_delta_log_evidence_vs_glm'] for row in rows])
    labels = [row['model'] for row in rows]
    colors = [MODEL_COLORS[row['model_id']] for row in rows]
    y = np.arange(len(rows))
    n_subjects = rows[0]['n_subjects']

    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    subject_vals = deltas['delta_log_evidence_vs_glm'].to_numpy(float)
    max_right = float(np.nanmax(np.concatenate([means + sems, subject_vals])))
    min_left = float(np.nanmin(np.concatenate([means - sems, subject_vals])))
    span = max(1.0, max_right - min_left)
    if per_trial:
        xlim = (
            min(-0.25, min_left - 0.10 * span),
            max(0.25, max_right + 0.16 * span),
        )
    else:
        xlim = (
            min(-10.0, min_left - 0.10 * span),
            max(5.0, max_right + 0.16 * span),
        )

    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    fig.subplots_adjust(left=0.39, right=0.98, bottom=0.25, top=0.92)
    ax.barh(y, means, xerr=sems, color=colors, height=0.52, edgecolor='none',
            error_kw={'elinewidth': 0.8, 'capsize': 2.0, 'capthick': 0.8})
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    if per_trial:
        ax.set_xlabel(r'$\Delta$ log evidence / trial vs. GLM (shallow)')
    else:
        ax.set_xlabel(r'$\Delta$ log evidence vs. GLM (shallow) (nats)')
    ax.set_xlim(*xlim)
    ax.grid(axis='x', color='#D9D9D9', linewidth=0.6)
    ax.set_axisbelow(True)

    rng = np.random.default_rng(0)
    for yi, model_id in zip(y, MODEL_ORDER):
        vals = deltas.loc[
            deltas['model_id'] == model_id,
            'delta_log_evidence_vs_glm',
        ].to_numpy(float)
        jitter = rng.normal(0.0, 0.035, size=vals.size)
        ax.scatter(vals, yi + jitter, s=6, color='#222222', alpha=0.38,
                   linewidths=0, zorder=3)

    for yi, mean, sem, model_id in zip(y, means, sems, MODEL_ORDER):
        if model_id == 'glm_shallow_zero_mean':
            continue
        if abs(mean) < 1e-9:
            text = '+0'
            x = 0.03 * span
            ha = 'left'
            color = '#222222'
        elif mean < 0 and abs(mean) > 0.15 * span:
            text = f'{mean:+.3f}' if per_trial else f'{mean:+.1f}'
            x = mean + 0.035 * span
            ha = 'left'
            color = 'white'
        elif mean > 0 and abs(mean) > 0.15 * span:
            text = f'{mean:+.3f}' if per_trial else f'{mean:+.1f}'
            x = mean - 0.035 * span
            ha = 'right'
            color = 'white'
        else:
            text = f'{mean:+.3f}' if per_trial else f'{mean:+.1f}'
            x = mean + np.sign(mean) * (0.035 * span)
            ha = 'left' if mean >= 0 else 'right'
            color = '#222222'
        bbox = None if color == 'white' else {
            'facecolor': 'white',
            'edgecolor': 'none',
            'alpha': 0.82,
            'pad': 0.15,
        }
        ax.text(x, yi, text, va='center', ha=ha, fontsize=6.6, color=color,
                bbox=bbox, zorder=4)

    animal_label = 'animal' if n_subjects == 1 else 'animals'
    ax.text(
        0.98,
        0.96,
        f'n = {n_subjects} {animal_label}',
        transform=ax.transAxes,
        va='top',
        ha='right',
        fontsize=6.5,
        color='#555555',
        bbox={
            'facecolor': 'white',
            'edgecolor': 'none',
            'alpha': 0.82,
            'pad': 0.2,
        },
    )

    fig.savefig(output_stem.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(output_stem.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(output_stem.with_suffix('.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    args.summary_csv = _resolve_repo_path(args.summary_csv)
    args.output_stem = _resolve_repo_path(args.output_stem)
    deltas = load_subject_deltas(args.summary_csv, per_trial=args.per_trial)
    rows = summarize(deltas)
    values_path, subject_path = save_values(deltas, rows, args.output_stem)
    plot(deltas, rows, args.output_stem, per_trial=args.per_trial)
    print(f'Wrote {_display_path(values_path)}')
    print(f'Wrote {_display_path(subject_path)}')
    print(f'Wrote {_display_path(args.output_stem.with_suffix(".pdf"))}')
    print(f'Wrote {_display_path(args.output_stem.with_suffix(".svg"))}')
    print(f'Wrote {_display_path(args.output_stem.with_suffix(".png"))}')
    for row in rows:
        print(
            f"{row['model']}: "
            f"mean delta={row['mean_delta_log_evidence_vs_glm']:+.3f} "
            f"+/- {row['sem_delta_log_evidence_vs_glm']:.3f} SEM"
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
