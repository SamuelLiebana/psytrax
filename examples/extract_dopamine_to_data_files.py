"""Build psytrax-format per-mouse data files from the public CSVs.

Reads:
    data/long_term_learning_dataset_preprocessed_behaviour_all.csv         (RT, choice, …)
    data/long_term_learning_dataset_preprocessed_photometry_zscore_…_all.csv (fStimulusOnset_*)

Produces, for every mouse that has at least one DLS recording:

    data/<mouse>_data.npy

When a mouse has both Left DLS and Right DLS recordings, the per-trial
peak dopamine is averaged across the two hemispheres (per `nanmean`, so
single-hemisphere trials still contribute).

Each file has the standard psytrax dict layout:

    {
        'inputs':           {'c': <signed contrast, shape (N,)>},
        'responses':        <0/1 right choice, shape (N,)>,
        'times':            <RT in seconds, shape (N,)>,
        'session_lengths':  <int array of trials per session>,
        'dopamine':         <peak of MA(window=10) of fStimulusOnset
                             over time ∈ [0.2, 0.35]s, shape (N,)>,
        't_nd':             <0.05 — same convention as the existing files>,
    }

Filtering, mirroring `dls_dopamine.ipynb`:
  * only expert sessions (``isExpertMouse == 1``)
  * 0.05s < RT < 1.5s
  * choice ∈ {'Left', 'Right'} (NoGo trials dropped)
  * trial must have a valid (non-NaN) dopamine peak in [0.2, 0.35]s
  * recording region ∈ {'Left DLS', 'Right DLS'} by default; left and
    right DLS are nan-mean averaged per trial.  Pass ``--regions all`` to
    keep every region in the CSV (still averaged across all matching rows
    per trial), or ``--regions "Left DLS"`` to restrict to one hemisphere.

By default the per-trial dopamine peaks are normalised by the
**per-session, per-region zero-contrast peak** (matching
`zero_contrast_normalisation_aligned` in the reference notebook):

    baseline_session = mean of fTimewarped[80..84] across 0-contrast
                       rewarded trials in this (region, session)
    peak_session     = mean of the top-3 fTimewarped values at
                       timewarped_time ≥ 82 across the same trials
    DA_normalised    = DA_raw / (peak_session − baseline_session)

This rescales each session's stimulus-aligned peak by that session's
free-reward outcome response amplitude, so the values are roughly
comparable across mice and sessions.  Pass ``--norm-mode percentile`` to
use a generic 5th/95th percentile min-max instead, or ``--norm-mode
none`` to keep the raw z-scored values.

Run from the repo root:

    python -m examples.extract_dopamine_to_data_files

Output files land directly in ``data/<mouse>_data.npy``, overwriting any
previous per-mouse file for that mouse.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / 'data'
# All per-mouse files now live directly under data/ (the with_dopamine
# subfolder was retired once every file gained a dopamine field).
OUT_DIR   = DATA_DIR

BEHAV_CSV = DATA_DIR / 'long_term_learning_dataset_preprocessed_behaviour_all.csv'
PHOTOM_CSV = (
    DATA_DIR
    / 'long_term_learning_dataset_preprocessed_photometry_zscore_akam_corrected_with_timewarped_stimulusOnset_outcome_-0.5_1_all.csv'
)

# Filtering thresholds — match the dls_dopamine.ipynb conventions.
RT_MIN   = 0.05
RT_MAX   = 1.5
DA_TMIN  = 0.20
DA_TMAX  = 0.35
ROLL_WIN = 10
T_ND     = 0.05    # mirrors the existing per-mouse files

# Default region whitelist — DLS only.  aDMS, pDMS, NAc and any other
# regions in the CSV are skipped unless the user passes ``--regions``
# explicitly.  Pass ``--regions all`` to disable region filtering.
DEFAULT_REGIONS = ['Left DLS', 'Right DLS']


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--photom-chunksize', type=int, default=20000,
                        help='Rows per chunk when streaming the 3 GB photometry CSV.')
    parser.add_argument('--out-dir', type=str, default=str(OUT_DIR),
                        help='Where to write per-mouse .npy files.')
    parser.add_argument('--mice', type=str, nargs='*', default=None,
                        help='Optional whitelist of mouse IDs (e.g. DAP022 DAP039). '
                             'If omitted, every mouse with ≥1 valid trial is exported.')
    parser.add_argument('--regions', type=str, nargs='*',
                        default=DEFAULT_REGIONS,
                        help=(
                            'Region whitelist; defaults to DLS only '
                            f'({DEFAULT_REGIONS}). Pass --regions all to '
                            'keep every region in the CSV (aDMS, pDMS, NAc, …). '
                            'Quote strings containing spaces.'
                        ))
    parser.add_argument('--norm-mode', type=str, default='zero-contrast',
                        choices=['zero-contrast', 'percentile', 'none'],
                        help=('Normalisation strategy for the per-trial '
                              'dopamine peak.  zero-contrast (default): per '
                              '(session, region) divide by the peak free-reward '
                              'response on 0-contrast rewarded trials, '
                              'mirroring src/load_data.py:'
                              'zero_contrast_normalisation_aligned.  '
                              'percentile: per-mouse 5th/95th percentile '
                              'min-max into [0, 1].  none: keep raw z-scored '
                              'values.'))
    parser.add_argument('--normalise-percentiles', type=float, nargs=2,
                        default=(5.0, 95.0), metavar=('LO', 'HI'),
                        help='Lower / upper percentiles for --norm-mode '
                             'percentile.  Default: 5 95.')
    return parser.parse_args()


# Time-warped indices used by the zero-contrast normalisation, mirroring
# zero_contrast_normalisation_aligned.  Baseline = mean over 80..84
# (notebook used a strict (>79, <85) filter).  Peak = top-3 mean over
# 82..142 (notebook used >=82).
ZC_BASELINE_TIMES = list(range(80, 85))           # 80, 81, 82, 83, 84
ZC_MAX_TIMES      = list(range(82, 143))          # 82 .. 142
ZC_TOPK           = 3


def _normalise_per_mouse(da: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
    """Min-max normalise to [0, 1] using percentile cutoffs (clipped).

    Robust to outliers: anything below the lo-percentile maps to 0, anything
    above the hi-percentile maps to 1, and the rest is linearly scaled.
    """
    finite = da[np.isfinite(da)]
    if finite.size < 2:
        return da
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros_like(da)
    out = (da - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _zero_contrast_normalise(photom: pd.DataFrame) -> pd.DataFrame:
    """Apply per-(region, expRef) zero-contrast normalisation in-place.

    Mirrors src/load_data.py:zero_contrast_normalisation_aligned —
    rescales each trial's stimulus-aligned dopamine peak by that
    session's free-reward outcome amplitude:

        baseline = mean of fTimewarped[80..84] across 0-contrast
                   rewarded trials in this (region, expRef)
        peak     = mean of top-3 fTimewarped[82..142] over the same trials
                   (top-3 done per trial, then averaged across trials)
        DA_norm  = DA_raw / (peak − baseline)

    Sessions whose denominator is non-positive or undefined are dropped
    (their dopamine becomes NaN and is filtered out downstream).
    """
    print('[normalise] zero-contrast: estimating per-(region, expRef) factors')
    rewarded_zero = (
        (photom['contrast'].astype(float) == 0.0)
        & (photom['feedback'].astype(str) == 'Rewarded')
    )
    factors = (
        photom.loc[rewarded_zero]
        .groupby(['region', 'expRef'], sort=False)
        .agg(
            baseline=('zc_baseline_mean', lambda v: float(np.nanmean(v))),
            peak    =('zc_max_top3_mean',  lambda v: float(np.nanmean(v))),
            n_zc    =('zc_baseline_mean', 'size'),
        )
        .reset_index()
    )
    factors['denom'] = factors['peak'] - factors['baseline']
    n_factors = len(factors)
    n_valid   = int(np.sum(factors['denom'] > 0))
    print(f'[normalise]   {n_valid}/{n_factors} (region, session) pairs have '
          f'a usable normaliser (denom > 0)')

    photom = photom.merge(
        factors[['region', 'expRef', 'denom', 'n_zc']],
        on=['region', 'expRef'], how='left',
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = photom['denom'].astype(float).to_numpy()
        ok = (denom > 0) & np.isfinite(denom)
        photom['dopamine'] = np.where(
            ok, photom['dopamine'].astype(float) / np.where(ok, denom, 1.0),
            np.nan,
        )
    n_dropped = int(np.sum(~ok))
    if n_dropped:
        print(f'[normalise]   dropped {n_dropped:,} trials whose '
              f'(region, session) had no valid normaliser')
    # Strip the helper columns before returning so they don't leak into
    # downstream stages.
    return photom.drop(columns=[
        'zc_baseline_mean', 'zc_max_top3_mean',
        'contrast', 'feedback', 'denom', 'n_zc',
    ])


# ---------------------------------------------------------------------------
# Behavioural CSV → trimmed dataframe with rt / signed contrast / 0-or-1 choice
# ---------------------------------------------------------------------------

def _load_behaviour() -> pd.DataFrame:
    """Return one row per surviving (expRef, trialNumber) trial."""
    print(f'[behav] reading {BEHAV_CSV}')
    cols = [
        'expRef', 'trialNumber',
        'contrastLeft', 'contrastRight',
        'choice', 'choiceCompleteTime', 'stimulusOnsetTime',
        'isExpertMouse',
    ]
    df = pd.read_csv(BEHAV_CSV, usecols=cols, low_memory=False)
    print(f'[behav] read {len(df):,} rows')

    df = df[df['isExpertMouse'] == 1].copy()
    df['rt'] = df['choiceCompleteTime'] - df['stimulusOnsetTime']
    df = df[(df['rt'] > RT_MIN) & (df['rt'] < RT_MAX)]
    df = df[df['choice'].isin(['Left', 'Right'])]
    df['c'] = df['contrastRight'].astype(float) - df['contrastLeft'].astype(float)
    df['r'] = (df['choice'] == 'Right').astype(float)

    df = df[['expRef', 'trialNumber', 'c', 'r', 'rt']].reset_index(drop=True)
    print(f'[behav] {len(df):,} trials kept after expert / RT / choice filtering '
          f'across {df["expRef"].nunique()} sessions')
    return df


# ---------------------------------------------------------------------------
# Photometry CSV → per-trial dopamine peak
# ---------------------------------------------------------------------------

_FSTIM_RE = re.compile(r'^fStimulusOnset(-?\d+(?:\.\d+)?)$')


def _select_dopamine_columns(header: list[str]) -> tuple[list[str], np.ndarray]:
    """Return the fStimulusOnset columns whose timestamp lies in the analysis window.

    Output is sorted ascending by timestamp so the rolling-mean operates
    along increasing time.
    """
    pairs = []
    for col in header:
        m = _FSTIM_RE.match(col)
        if m is None:
            continue
        t = float(m.group(1))
        if DA_TMIN <= t <= DA_TMAX:
            pairs.append((t, col))
    pairs.sort()
    cols = [c for _, c in pairs]
    times = np.array([t for t, _ in pairs], dtype=float)
    return cols, times


def _peak_per_row(values: np.ndarray, window: int) -> np.ndarray:
    """Per-row peak of a `window`-sample rolling mean.

    ``values`` has shape (n_trials, n_timepoints). Returns shape (n_trials,).
    NaN inputs are handled gracefully (a trial with all-NaN timepoints
    yields NaN).
    """
    if values.size == 0:
        return np.full(values.shape[0], np.nan)
    n_t = values.shape[1]
    if n_t < window:
        # Not enough samples for a full window — fall back to the mean
        # over whatever we have, which is what pandas' rolling(min_periods=1)
        # would also do for the last sample.
        return np.nanmean(values, axis=1)
    cumulative = np.nancumsum(values, axis=1)
    cumulative = np.concatenate(
        [np.zeros((values.shape[0], 1), dtype=cumulative.dtype), cumulative],
        axis=1,
    )
    valid_counts = np.cumsum(np.isfinite(values), axis=1)
    valid_counts = np.concatenate(
        [np.zeros((values.shape[0], 1), dtype=int), valid_counts],
        axis=1,
    )
    # Number of *valid* samples in each rolling window of size `window`.
    win_valid = valid_counts[:, window:] - valid_counts[:, :-window]
    win_sum   = cumulative[:, window:] - cumulative[:, :-window]
    with np.errstate(divide='ignore', invalid='ignore'):
        win_mean = np.where(win_valid > 0, win_sum / np.maximum(win_valid, 1), np.nan)
    return np.nanmax(win_mean, axis=1)


def _load_dopamine_peaks(photom_chunksize: int,
                         need_normalisation_cols: bool) -> pd.DataFrame:
    """Return one row per (expRef, trialNumber, region) with:
      * dopamine: peak of MA(window=10) of fStimulusOnset over [0.2, 0.35]s
      * (when ``need_normalisation_cols``) zc_baseline_sum, zc_baseline_count,
        zc_max_top1..top3, contrast, feedback — used to compute the
        per-(region, expRef) zero-contrast normalisation factors.
    """
    print(f'[photom] streaming {PHOTOM_CSV}')
    header = pd.read_csv(PHOTOM_CSV, nrows=0).columns.tolist()
    da_cols, da_times = _select_dopamine_columns(header)
    if not da_cols:
        raise SystemExit(
            f'[photom] no fStimulusOnset columns found in [{DA_TMIN}, {DA_TMAX}]s — '
            'check the CSV layout'
        )
    print(f'[photom] using {len(da_cols)} stim-onset timepoints '
          f'(t = [{da_times.min():.3f}, {da_times.max():.3f}]s)')

    keep_cols = ['expRef', 'trialNumber', 'region', *da_cols]
    base_cols = max_cols = []
    if need_normalisation_cols:
        keep_cols += ['contrast', 'feedback']
        base_cols = [f'fTimewarped{t:.1f}' for t in ZC_BASELINE_TIMES]
        max_cols  = [f'fTimewarped{t:.1f}' for t in ZC_MAX_TIMES]
        missing = [c for c in base_cols + max_cols if c not in header]
        if missing:
            raise SystemExit(
                f'[photom] zero-contrast normalisation requires fTimewarped '
                f'columns {missing[:3]}{" …" if len(missing) > 3 else ""} '
                f'which are missing from the CSV.  Use --norm-mode percentile '
                'or --norm-mode none.'
            )
        keep_cols += base_cols + max_cols
        print(f'[photom] also pulling {len(base_cols)} baseline + '
              f'{len(max_cols)} peak fTimewarped columns for zero-contrast '
              'normalisation')

    chunks = []
    for i, ch in enumerate(pd.read_csv(
            PHOTOM_CSV, usecols=keep_cols, chunksize=photom_chunksize,
            low_memory=False)):
        vals = ch[da_cols].to_numpy(dtype=float, copy=False)
        peak = _peak_per_row(vals, ROLL_WIN)
        out = {
            'expRef':       ch['expRef'].to_numpy(),
            'trialNumber':  ch['trialNumber'].to_numpy(),
            'region':       ch['region'].to_numpy(),
            'dopamine':     peak,
        }
        if need_normalisation_cols:
            base_arr = ch[base_cols].to_numpy(dtype=float, copy=False)
            max_arr  = ch[max_cols].to_numpy(dtype=float, copy=False)
            out['zc_baseline_mean']  = np.nanmean(base_arr, axis=1)
            # Per-trial top-3 mean of the post-event window so the
            # downstream session aggregation can recompute the notebook's
            # `top_k(3).mean()` exactly.
            sorted_max = np.sort(max_arr, axis=1)
            top3 = sorted_max[:, -ZC_TOPK:]
            out['zc_max_top3_mean'] = np.nanmean(top3, axis=1)
            out['contrast'] = ch['contrast'].to_numpy()
            out['feedback'] = ch['feedback'].to_numpy()
        chunks.append(pd.DataFrame(out))
        if (i + 1) % 5 == 0:
            print(f'[photom]   {(i + 1) * photom_chunksize:,} rows processed…')
    df = pd.concat(chunks, ignore_index=True)
    print(f'[photom] {len(df):,} (expRef, trial, region) rows total')
    return df


# ---------------------------------------------------------------------------
# Joining + per-(mouse, region) file writing
# ---------------------------------------------------------------------------

def _mouse_id_from_expref(expref: str) -> str | None:
    """Extract the mouse ID (e.g. 'DAP039') from '2021-07-12_1_DAP039'."""
    parts = expref.split('_')
    return parts[-1] if parts else None


def _write_one(out_dir: Path, mouse: str, trials: pd.DataFrame,
               regions_used: list[str]) -> str:
    """Write a single `<mouse>_data.npy` and return its filename."""
    # Per-session order — sort by expRef (date-encoded) then trialNumber.
    trials = trials.sort_values(['expRef', 'trialNumber']).reset_index(drop=True)
    session_lengths = (
        trials.groupby('expRef', sort=False).size().to_numpy(dtype=int)
    )
    data = {
        'inputs':          {'c': trials['c'].to_numpy(dtype=float)},
        'responses':       trials['r'].to_numpy(dtype=float),
        'times':           trials['rt'].to_numpy(dtype=float),
        'session_lengths': session_lengths,
        'dopamine':        trials['dopamine'].to_numpy(dtype=float),
        't_nd':            np.float64(T_ND),
    }
    path = out_dir / f'{mouse}_data.npy'
    np.save(path, data, allow_pickle=True)
    print(f'  → {path.name}: {len(trials)} trials, '
          f'{len(session_lengths)} sessions, '
          f'regions averaged = {regions_used}, '
          f'DA mean={np.nanmean(data["dopamine"]):.3f} '
          f'(min={np.nanmin(data["dopamine"]):.3f}, '
          f'max={np.nanmax(data["dopamine"]):.3f})')
    return path.name


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not BEHAV_CSV.exists():
        print(f'ERROR: {BEHAV_CSV} not found', file=sys.stderr)
        return 1
    if not PHOTOM_CSV.exists():
        print(f'ERROR: {PHOTOM_CSV} not found', file=sys.stderr)
        return 1

    behav = _load_behaviour()
    need_zc = (args.norm_mode == 'zero-contrast')
    photom = _load_dopamine_peaks(args.photom_chunksize,
                                  need_normalisation_cols=need_zc)

    # Apply zero-contrast normalisation BEFORE region filtering so that
    # mice with photometry from non-DLS regions can still contribute their
    # 0-contrast trials if needed.  (Currently the function uses every
    # available row in each (region, expRef) bucket, so post-filtering is
    # equivalent — but doing it now keeps the order analogous to the
    # notebook.)
    if need_zc:
        photom = _zero_contrast_normalise(photom)

    # Inner join: every retained trial must have both behaviour and DA.
    merged = photom.merge(behav, on=['expRef', 'trialNumber'], how='inner')
    merged = merged[np.isfinite(merged['dopamine'])]
    print(f'[merge] {len(merged):,} (trial × region) rows after inner join '
          f'and DA-finite filter')

    # Map expRef → mouse ID once.
    merged['mouse'] = merged['expRef'].map(_mouse_id_from_expref)

    if args.mice:
        merged = merged[merged['mouse'].isin(args.mice)]
        print(f'[filter] mouse whitelist applied: {len(merged):,} rows kept')

    # ``--regions all`` (case-insensitive) disables region filtering entirely.
    region_all = (
        args.regions is not None
        and len(args.regions) == 1
        and args.regions[0].lower() == 'all'
    )
    if args.regions and not region_all:
        merged = merged[merged['region'].isin(args.regions)]
        print(f'[filter] region whitelist {args.regions} applied: '
              f'{len(merged):,} rows kept')
    elif region_all:
        print(f'[filter] --regions all → keeping every region '
              f'({merged["region"].nunique()} unique values, '
              f'{len(merged):,} rows)')

    if merged.empty:
        print('No rows match — nothing written.')
        return 1

    # Collapse multiple regions per (mouse, expRef, trialNumber) → a single
    # row whose dopamine value is the nan-mean across regions.  Behavioural
    # fields (c, r, rt) are constant across hemispheres so we just take the
    # first.
    print('[merge] averaging dopamine across regions per trial…')
    grouped = (
        merged
        .groupby(['mouse', 'expRef', 'trialNumber'], sort=False)
        .agg(
            c=('c', 'first'),
            r=('r', 'first'),
            rt=('rt', 'first'),
            dopamine=('dopamine', lambda v: float(np.nanmean(v))),
        )
        .reset_index()
    )
    grouped = grouped[np.isfinite(grouped['dopamine'])]
    print(f'[merge] {len(grouped):,} trials after region averaging')

    # Per-mouse list of regions actually present (for the log line).
    regions_per_mouse = (
        merged.groupby('mouse')['region'].unique().apply(sorted).to_dict()
    )

    if args.norm_mode == 'percentile':
        lo_pct, hi_pct = args.normalise_percentiles
        print(f'[normalise] per-mouse min-max to [0, 1] using '
              f'{lo_pct}/{hi_pct} percentiles')
        for mouse, idx in grouped.groupby('mouse').groups.items():
            da = grouped.loc[idx, 'dopamine'].to_numpy(dtype=float)
            grouped.loc[idx, 'dopamine'] = _normalise_per_mouse(
                da, lo_pct, hi_pct,
            )
    elif args.norm_mode == 'none':
        print('[normalise] skipped (raw z-scored values written).')
    else:
        # zero-contrast normalisation already applied per (region, expRef)
        # before the region averaging, so nothing extra to do here.
        print('[normalise] zero-contrast factors already applied per (region, expRef)')

    print(f'[write] writing per-mouse files into {out_dir}')
    written = []
    for mouse, sub in grouped.groupby('mouse', sort=True):
        if not mouse:
            continue
        if len(sub) < 50:
            # 50 trials is a sane minimum for any psytrax fit.
            continue
        written.append(_write_one(
            out_dir, mouse, sub, regions_per_mouse.get(mouse, [])
        ))

    print(f'\n[done] wrote {len(written)} files to {out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
