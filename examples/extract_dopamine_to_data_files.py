"""Build psytrax-format per-mouse data files from the public CSVs.

Reads:
    data/long_term_learning_dataset_preprocessed_behaviour_all.csv         (RT, choice, …)
    data/long_term_learning_dataset_preprocessed_photometry_zscore_…_all.csv (fStimulusOnset_*)

Produces, for every (mouse, region) pair that has both behavioural and
photometry rows:

    data/with_dopamine/<mouse>__<region>_data.npy

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

Run from the repo root:

    python -m examples.extract_dopamine_to_data_files

The script writes to a separate folder (``data/with_dopamine/``) so the
existing per-mouse files in ``data/`` are untouched until you've eyeballed
the new ones and decided to swap them in.
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
OUT_DIR   = DATA_DIR / 'with_dopamine'

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
    parser.add_argument('--regions', type=str, nargs='*', default=None,
                        help='Optional whitelist of regions (e.g. "Left DLS"). '
                             'Pass quoted strings if they contain spaces.')
    return parser.parse_args()


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


def _load_dopamine_peaks(photom_chunksize: int) -> pd.DataFrame:
    """Return one row per (expRef, trialNumber, region) with the peak DA value."""
    print(f'[photom] streaming {PHOTOM_CSV}')
    header = pd.read_csv(PHOTOM_CSV, nrows=0).columns.tolist()
    da_cols, da_times = _select_dopamine_columns(header)
    if not da_cols:
        raise SystemExit(
            f'[photom] no fStimulusOnset columns found in [{DA_TMIN}, {DA_TMAX}]s — '
            'check the CSV layout'
        )
    print(f'[photom] using {len(da_cols)} timepoints '
          f'(t = [{da_times.min():.3f}, {da_times.max():.3f}]s)')

    keep_cols = ['expRef', 'trialNumber', 'region', *da_cols]
    chunks = []
    for i, ch in enumerate(pd.read_csv(
            PHOTOM_CSV, usecols=keep_cols, chunksize=photom_chunksize,
            low_memory=False)):
        vals = ch[da_cols].to_numpy(dtype=float, copy=False)
        peak = _peak_per_row(vals, ROLL_WIN)
        chunks.append(pd.DataFrame({
            'expRef':       ch['expRef'].to_numpy(),
            'trialNumber':  ch['trialNumber'].to_numpy(),
            'region':       ch['region'].to_numpy(),
            'dopamine':     peak,
        }))
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


def _safe_region_token(region: str) -> str:
    """Filename-safe version of a region label."""
    return re.sub(r'[^A-Za-z0-9]+', '_', str(region)).strip('_')


def _write_one(out_dir: Path, mouse: str, region: str,
               trials: pd.DataFrame) -> str:
    """Write a single `<mouse>__<region>_data.npy` and return its filename."""
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
    fname = f'{mouse}__{_safe_region_token(region)}_data.npy'
    path = out_dir / fname
    np.save(path, data, allow_pickle=True)
    print(f'  → {path.name}: {len(trials)} trials, '
          f'{len(session_lengths)} sessions, '
          f'DA mean={np.nanmean(data["dopamine"]):.3f} '
          f'(min={np.nanmin(data["dopamine"]):.3f}, '
          f'max={np.nanmax(data["dopamine"]):.3f})')
    return fname


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
    photom = _load_dopamine_peaks(args.photom_chunksize)

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
    if args.regions:
        merged = merged[merged['region'].isin(args.regions)]
        print(f'[filter] region whitelist applied: {len(merged):,} rows kept')

    if merged.empty:
        print('No rows match — nothing written.')
        return 1

    print(f'[write] writing per-(mouse, region) files into {out_dir}')
    written = []
    for (mouse, region), sub in merged.groupby(['mouse', 'region'], sort=True):
        if not mouse or not region:
            continue
        if len(sub) < 50:
            # 50 trials is a sane minimum for any psytrax fit.
            continue
        written.append(_write_one(out_dir, mouse, region, sub))

    print(f'\n[done] wrote {len(written)} files to {out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
