"""Generate documentation figures from the bundled DAP011 race-model fit.

Usage:
    conda run --no-capture-output -n psytrax python examples/generate_dap011_docs_figures.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm as _sp_norm


REPO_DIR = os.path.dirname(os.path.dirname(__file__))
FIT_PATH = os.path.join(REPO_DIR, "example_fits", "DAP011_race_fit.npy")
OUT_DIR = os.path.join(REPO_DIR, "examples", "assets")

COLORS = ['#4e9af1', '#f1a44e', '#4ef17a', '#f14e7a', '#c44ef1']


def _ig_cdf(thr, drift, v, t):
    t = np.maximum(t, 1e-12)
    a = _sp_norm.cdf((drift * t - thr) / np.sqrt(v * t))
    log_b = 2.0 * thr * (drift / v) + _sp_norm.logcdf(-(drift * t + thr) / np.sqrt(v * t))
    return np.clip(a + np.exp(log_b), 0.0, 1.0)


def _ig_pdf(thr, drift, v, t):
    t = np.maximum(t, 1e-12)
    return thr / np.sqrt(2 * np.pi * v * t ** 3) * np.exp(-(thr - drift * t) ** 2 / (2 * v * t))


def _race_curves(params_window, param_names, c_grid, fixed_params, t_max=30.0, n_t=2000):
    mp = np.mean(params_window, axis=1)
    idx = {name: i for i, name in enumerate(param_names)}
    wr = mp[idx['wr']]
    wl = mp[idx['wl']]
    br = mp[idx['br']]
    bl = mp[idx['bl']]
    z = mp[idx['z']]
    sig_i = float(fixed_params['sig_i'])

    t_grid = np.linspace(1e-4, t_max, n_t)
    p_rights = np.zeros(len(c_grid))
    mean_rts = np.zeros(len(c_grid))

    for i, c in enumerate(c_grid):
        d1 = wr * max(c, 0.0) + br
        d2 = wl * max(-c, 0.0) + bl
        v1 = float(wr ** 2 * sig_i ** 2 + 1.0)
        v2 = float(wl ** 2 * sig_i ** 2 + 1.0)
        f1 = _ig_pdf(z, d1, v1, t_grid)
        f2 = _ig_pdf(z, d2, v2, t_grid)
        F1 = _ig_cdf(z, d1, v1, t_grid)
        F2 = _ig_cdf(z, d2, v2, t_grid)
        p_rights[i] = np.clip(np.trapezoid(f1 * (1 - F2), t_grid), 0.0, 1.0)
        mean_rts[i] = max(np.trapezoid((1 - F1) * (1 - F2), t_grid), 0.0)

    return p_rights, mean_rts


def _shared_ylim(series_list, pad_frac=0.05, min_pad=0.05):
    vals = []
    for series in series_list:
        arr = np.asarray(series, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(arr)
    if not vals:
        return None
    all_vals = np.concatenate(vals)
    ymin = float(np.min(all_vals))
    ymax = float(np.max(all_vals))
    span = ymax - ymin
    pad = max(min_pad, span * pad_frac)
    if span == 0:
        pad = max(min_pad, abs(ymin) * pad_frac, 0.01)
    return ymin - pad, ymax + pad


def _style_ax(ax, xlabel=None, ylabel=None, title=None):
    ax.set_facecolor('white')
    ax.grid(True, color='#ececec', linewidth=0.8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11, pad=8)


def _save_trajectories(result):
    params = result['params']
    param_names = result['param_names']
    W_std = result.get('hess_info', {}).get('W_std')
    dat = result['data']
    n_trials = params.shape[1]
    trials = np.arange(n_trials)
    day_lengths = dat.get('dayLength') if dat.get('dayLength') is not None else np.array([])
    boundaries = np.cumsum(day_lengths).astype(int) if len(day_lengths) else np.array([], dtype=int)

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    for k, (ax, name) in enumerate(zip(axes, param_names)):
        color = COLORS[k % len(COLORS)]
        _style_ax(ax, ylabel='Value', title=name)
        ax.plot(trials, params[k], color=color, lw=1.0)
        if W_std is not None:
            ax.fill_between(trials, params[k] - W_std[k], params[k] + W_std[k],
                            color=color, alpha=0.18)
        for b in boundaries[:-1]:
            ax.axvline(b, color='#b8b8b8', lw=0.8, alpha=0.8, ls='--')

    axes[4].set_xlabel('Trial')
    axes[5].set_visible(False)
    fig.suptitle('DAP011 race-model parameter trajectories', fontsize=15, y=0.99)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'dap011_race_trajectories.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def _save_evolution_curves(result):
    params = result['params']
    param_names = result['param_names']
    dat = result['data']
    fixed_params = result.get('fixed_params', {})

    c_data = dat['inputs']['c']
    r_data = dat['r']
    t_data = dat['T']
    n_trials = params.shape[1]
    n_win = 4
    edges = np.linspace(0, n_trials, n_win + 1, dtype=int)
    contrasts_unique = np.unique(c_data)
    c_grid = np.linspace(contrasts_unique.min(), contrasts_unique.max(), 100)

    fig_p, axes_p = plt.subplots(2, 2, figsize=(11, 8))
    fig_c, axes_c = plt.subplots(2, 2, figsize=(11, 8))

    chrono_panel_data = []
    y_series = []

    for wi, ax in enumerate(axes_p.flat):
        t0, t1 = int(edges[wi]), int(edges[wi + 1])
        mask = np.zeros(n_trials, dtype=bool)
        mask[t0:t1] = True

        c_win = c_data[mask]
        r_win = r_data[mask]
        rt_win = t_data[mask]
        c_uniq_win = np.unique(c_win)
        p_win = np.array([r_win[c_win == cv].mean() for cv in c_uniq_win])
        rt_win_mean = np.array([rt_win[c_win == cv].mean() for cv in c_uniq_win])
        n_pts = np.array([np.sum(c_win == cv) for cv in c_uniq_win])
        p_model, rt_model = _race_curves(params[:, t0:t1], param_names, c_grid, fixed_params=fixed_params)

        _style_ax(ax, xlabel='Signed contrast', ylabel='P(right)', title=f'Trials {t0 + 1}-{t1}')
        ax.scatter(c_uniq_win, p_win, s=[max(12, n / 5) for n in n_pts], color='black', zorder=3)
        ax.plot(c_grid, p_model, color='#4e9af1', lw=2.2)
        ax.axhline(0.5, color='#b8b8b8', lw=0.8, ls='--')
        ax.axvline(0, color='#b8b8b8', lw=0.8, ls='--')
        ax.set_ylim(0, 1)

        chrono_panel_data.append((t0, t1, c_uniq_win, rt_win_mean, n_pts, rt_model))
        y_series.extend([rt_win_mean, rt_model])

    shared_ylim = _shared_ylim(y_series)

    for ax, (t0, t1, c_uniq_win, rt_win_mean, n_pts, rt_model) in zip(axes_c.flat, chrono_panel_data):
        _style_ax(ax, xlabel='Signed contrast', ylabel='Mean RT (s)', title=f'Trials {t0 + 1}-{t1}')
        ax.scatter(c_uniq_win, rt_win_mean, s=[max(12, n / 5) for n in n_pts], color='black', zorder=3)
        ax.plot(c_grid, rt_model, color='#4e9af1', lw=2.2)
        ax.axvline(0, color='#b8b8b8', lw=0.8, ls='--')
        if shared_ylim is not None:
            ax.set_ylim(*shared_ylim)

    fig_p.suptitle('DAP011 psychometric evolution over learning', fontsize=15, y=0.98)
    fig_p.tight_layout()
    fig_p.savefig(os.path.join(OUT_DIR, 'dap011_race_psychometric.png'), dpi=180, bbox_inches='tight')
    plt.close(fig_p)

    fig_c.suptitle('DAP011 chronometric evolution over learning', fontsize=15, y=0.98)
    fig_c.tight_layout()
    fig_c.savefig(os.path.join(OUT_DIR, 'dap011_race_chronometric.png'), dpi=180, bbox_inches='tight')
    plt.close(fig_c)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    result = np.load(FIT_PATH, allow_pickle=True).item()
    _save_trajectories(result)
    _save_evolution_curves(result)
    print(f"Saved PNGs to {OUT_DIR}")


if __name__ == '__main__':
    main()
