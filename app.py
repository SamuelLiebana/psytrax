"""psytrax web app — instructions, fit model, result visualiser.

Run with:  streamlit run app.py
"""

import io
import os
import threading
import queue
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm as _sp_norm


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

if hasattr(np, "trapezoid"):
    _trapezoid = np.trapezoid
else:
    _trapezoid = np.trapz

_RACE_DYNAMIC_PARAMS = {'wr', 'wl', 'br', 'bl', 'z'}
_RACE_FULL_PARAMS = _RACE_DYNAMIC_PARAMS | {'sig_i'}
_DDM_EXACT_PARAMS = {'w', 'b', 'a'}
_DDM_APPROX_PARAMS = {'w', 'b', 'z'}
_RT_CURVE_FAMILIES = {'race', 'ddm_exact', 'ddm_approx'}

# ---------------------------------------------------------------------------
# Race-model dopamine readout defaults — pulled from the model module so the
# plotted analytic curves match whatever β / offset the model was fitted
# with.  Wrapped in try/except so app.py can still load against an older
# psytrax.models.race that pre-dates the dopamine code (e.g. a stale
# cached install on a hosted deployment); in that case we fall back to the
# documented defaults and the dopamine plots still render.
# ---------------------------------------------------------------------------
try:
    from psytrax.models.race import (
        DEFAULT_DA_BETA   as _DA_BETA_DEFAULT,
        DEFAULT_DA_OFFSET as _DA_OFFSET_DEFAULT,
    )
except ImportError:
    _DA_BETA_DEFAULT, _DA_OFFSET_DEFAULT = 2.0, 0.001


def _is_logistic(param_names):
    """Detect a logistic model from its parameter names.

    Single-input logistic has names ``['w', 'b']``; multi-input variants
    use ``['w_<key1>', 'w_<key2>', ..., 'b']``.
    """
    if not param_names or param_names[-1] != 'b':
        return False
    weights = param_names[:-1]
    if not weights:
        return False
    return all(w == 'w' or w.startswith('w_') for w in weights)


# ---------------------------------------------------------------------------
# Theme-aware colours
# ---------------------------------------------------------------------------

_PLOT_COLOURS = {
    'bg':         'white',
    'text':       'black',
    'spine':      '#cccccc',
    'legend_bg':  '#f5f5f5',
    'legend_edge':'#cccccc',
}


def _style_fig(fig):
    """Set figure background to the standard dark plot theme."""
    fig.patch.set_facecolor(_PLOT_COLOURS['bg'])
    return _PLOT_COLOURS


def _style_legend(ax, **kwargs):
    """Apply standard dark styling to an axis legend."""
    ax.legend(
        facecolor=_PLOT_COLOURS['legend_bg'],
        edgecolor=_PLOT_COLOURS['legend_edge'],
        labelcolor=_PLOT_COLOURS['text'],
        **kwargs,
    )


def _style_ax(ax, xlabel=None, ylabel=None, title=None, title_pad=None):
    ax.set_facecolor(_PLOT_COLOURS['bg'])
    ax.tick_params(colors=_PLOT_COLOURS['text'])
    for spine in ax.spines.values():
        spine.set_edgecolor(_PLOT_COLOURS['spine'])
    if xlabel:
        ax.set_xlabel(xlabel, color=_PLOT_COLOURS['text'])
    if ylabel:
        ax.set_ylabel(ylabel, color=_PLOT_COLOURS['text'])
    if title:
        kwargs = {}
        if title_pad is not None:
            kwargs['pad'] = title_pad
        ax.set_title(title, color=_PLOT_COLOURS['text'], **kwargs)


def _show_fig(fig, filename='figure.png', dpi=200):
    """Display a matplotlib figure as an image with a PNG download button,
    then close it.

    Using ``st.image`` instead of ``st.pyplot`` renders a real ``<img>`` tag
    so users can right-click → *Copy Image*.  The download button provides a
    high-resolution PNG export.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    png_bytes = buf.getvalue()

    st.image(png_bytes, use_container_width=True)
    st.download_button(
        label=f'Download {filename}',
        data=png_bytes,
        file_name=filename,
        mime='image/png',
    )
    plt.close(fig)


def _ig_cdf(thr, drift, v, t):
    """Inverse-Gaussian CDF (vectorised over t)."""
    t = np.maximum(t, 1e-12)
    A = _sp_norm.cdf((drift * t - thr) / np.sqrt(v * t))
    logB = 2.0 * thr * (drift / v) + _sp_norm.logcdf(-(drift * t + thr) / np.sqrt(v * t))
    return np.clip(A + np.exp(logB), 0.0, 1.0)


def _ig_pdf(thr, drift, v, t):
    """Inverse-Gaussian PDF (vectorised over t)."""
    t = np.maximum(t, 1e-12)
    return thr / np.sqrt(2 * np.pi * v * t ** 3) * np.exp(-(thr - drift * t) ** 2 / (2 * v * t))


def _is_mlp(param_names):
    return (param_names[-1] == 'b2' and
            any(p.startswith('W1_') for p in param_names))


_MAX_CURVE_STATES = 8


def _sample_curve_states(params_window, max_states=_MAX_CURVE_STATES):
    """Select representative parameter states from a window.

    Averaging predictions across a few trial states better reflects
    learning-rule fits than collapsing the whole window to a single mean
    parameter vector.
    """
    params_window = np.asarray(params_window, dtype=float)
    if params_window.ndim == 1:
        return params_window[:, None]
    if params_window.ndim != 2:
        raise ValueError(f'params_window must be 1D or 2D, got shape {params_window.shape}')
    n_states = params_window.shape[1]
    if n_states <= max_states:
        return params_window
    idx = np.unique(np.linspace(0, n_states - 1, max_states, dtype=int))
    return params_window[:, idx]


def _mlp_psychometric(params_vec, param_names, c_grid):
    """P(right|c) for the MLP model over a contrast grid."""
    params_vec = np.asarray(params_vec, dtype=float)
    n_W1 = sum(1 for p in param_names if p.startswith('W1_'))
    H    = sum(1 for p in param_names if p.startswith('b1_'))
    n_in = n_W1 // H
    W1 = params_vec[:n_W1].reshape(n_in, H)
    b1 = params_vec[n_W1:n_W1 + H]
    W2 = params_vec[n_W1 + H:n_W1 + 2 * H]
    b2 = params_vec[-1]

    p_right = np.zeros(len(c_grid))
    for i, c in enumerate(c_grid):
        x      = np.zeros(n_in)
        x[0]   = c                          # first input is always contrast
        h      = np.tanh(W1.T @ x + b1)
        logit  = W2 @ h + b2
        p_right[i] = 1.0 / (1.0 + np.exp(-logit))
    return p_right


def _race_curves(params_vec, param_names, c_grid, fixed_params=None, t_max=30.0, n_t=2000):
    """Compute P(right|c) and E[min(T_R,T_L)|c] for one race-model state."""
    params_vec = np.asarray(params_vec, dtype=float)
    idx = {name: i for i, name in enumerate(param_names)}
    wr  = float(params_vec[idx['wr']]);  wl  = float(params_vec[idx['wl']])
    br  = float(params_vec[idx['br']]);  bl  = float(params_vec[idx['bl']])
    z   = float(params_vec[idx['z']])
    if 'sig_i' in idx:
        si = float(params_vec[idx['sig_i']])
    elif fixed_params and 'sig_i' in fixed_params:
        si = float(fixed_params['sig_i'])
    else:
        raise KeyError("Race-model visualisation requires either a sig_i parameter row or fixed_params['sig_i'].")

    t_grid = np.linspace(1e-4, t_max, n_t)
    p_rights = np.zeros(len(c_grid))
    mean_rts = np.zeros(len(c_grid))

    for i, c in enumerate(c_grid):
        d1 = wr * max(c, 0.0) + br   # right-accumulator drift
        d2 = wl * max(-c, 0.0) + bl  # left-accumulator drift
        v1 = float(wr ** 2 * si ** 2 + 1.0)
        v2 = float(wl ** 2 * si ** 2 + 1.0)
        F1 = _ig_cdf(z, d1, v1, t_grid)
        F2 = _ig_cdf(z, d2, v2, t_grid)
        f1 = _ig_pdf(z, d1, v1, t_grid)
        # P(right) = ∫ f_R(t)·(1−F_L(t)) dt
        p_rights[i] = np.clip(_trapezoid(f1 * (1 - F2), t_grid), 0.0, 1.0)
        # E[min(T_R,T_L)] = ∫ (1−F_R)(1−F_L) dt
        mean_rts[i] = max(_trapezoid((1 - F1) * (1 - F2), t_grid), 0.0)

    return p_rights, mean_rts


def _ddm_exact_hit_prob(drift, boundary, start):
    """Upper-boundary hit probability for the exact DDM."""
    drift = np.asarray(drift, dtype=float)
    p_right = np.empty_like(drift, dtype=float)
    near_zero = np.isclose(drift, 0.0, atol=1e-8)
    pos = drift > 1e-8
    neg = drift < -1e-8

    p_right[near_zero] = start / boundary
    if np.any(pos):
        p_right[pos] = (
            -np.expm1(-2.0 * drift[pos] * start)
            / -np.expm1(-2.0 * drift[pos] * boundary)
        )
    if np.any(neg):
        u = -2.0 * drift[neg]
        p_right[neg] = (
            np.exp(u * (start - boundary))
            * (-np.expm1(-u * start))
            / -np.expm1(-u * boundary)
        )
    return np.clip(p_right, 0.0, 1.0)


def _ddm_exact_curves(params_vec, param_names, c_grid):
    """Psychometric + chronometric predictions for one exact-DDM state.

    The starting point is fixed at a/2 (unbiased) — see psytrax.models.ddm
    for the rationale (b/z degeneracy is removed by hardcoding z = 0.5).
    """
    params_vec = np.asarray(params_vec, dtype=float)
    idx = {name: i for i, name in enumerate(param_names)}
    w = float(params_vec[idx['w']])
    b = float(params_vec[idx['b']])
    a = max(float(params_vec[idx['a']]), 1e-6)
    z_abs = 0.5 * a   # unbiased start, matches the model

    drift = w * np.asarray(c_grid, dtype=float) + b
    p_right = _ddm_exact_hit_prob(drift, a, z_abs)

    mean_rts = np.empty_like(drift, dtype=float)
    near_zero = np.isclose(drift, 0.0, atol=1e-8)
    mean_rts[near_zero] = z_abs * (a - z_abs)
    mean_rts[~near_zero] = (a * p_right[~near_zero] - z_abs) / drift[~near_zero]
    mean_rts = np.where(np.isfinite(mean_rts), np.maximum(mean_rts, 0.0), np.nan)
    return p_right, mean_rts


def _ddm_approx_curves(params_vec, param_names, c_grid, n_t=2000):
    """Compute psychometric and chronometric predictions for one approx-DDM state."""
    params_vec = np.asarray(params_vec, dtype=float)
    idx = {name: i for i, name in enumerate(param_names)}
    w = float(params_vec[idx['w']])
    b = float(params_vec[idx['b']])
    z = max(float(params_vec[idx['z']]), 1e-6)

    drift = w * np.asarray(c_grid, dtype=float) + b
    finite_abs = np.abs(drift[np.isfinite(drift)])
    slowest_drift = max(float(np.min(finite_abs)) if finite_abs.size else 0.05, 0.05)
    t_max = max(10.0, 12.0 * z / slowest_drift, 12.0 * z * z)
    # Log spacing keeps resolution near the sharp early-time peak without
    # sacrificing coverage of long tails when drift is close to zero.
    t_grid = np.geomspace(1e-4, t_max, n_t)

    p_rights = np.zeros(len(c_grid))
    mean_rts = np.zeros(len(c_grid))

    for i, v in enumerate(drift):
        F_right = _ig_cdf(z, v, 1.0, t_grid)
        F_left = _ig_cdf(z, -v, 1.0, t_grid)
        f_right = _ig_pdf(z, v, 1.0, t_grid)
        p_rights[i] = np.clip(_trapezoid(f_right * (1.0 - F_left), t_grid), 0.0, 1.0)
        mean_rts[i] = max(_trapezoid((1.0 - F_right) * (1.0 - F_left), t_grid), 0.0)

    return p_rights, mean_rts


def _model_family_info(param_names, result=None):
    """Detect the model family and pull out any constant nuisance params.

    Constant params can come from two sources:
      - ``result['model_hyper']`` (new K=5 race fits — sig_i is an EB-estimated
        model-level hyperparameter).
      - ``result['fixed_params']`` (legacy K=5 fits saved from the old
        ``make_fixed_sig_i_model`` wrapper).
    Both are merged into a single ``fixed_params`` dict for the curve
    helpers, which only need the scalar value.
    """
    result = result or {}
    fixed_params = dict(result.get('fixed_params') or {})
    fixed_params.update({k: float(v) for k, v in (result.get('model_hyper') or {}).items()})
    param_set = set(param_names)
    if param_set == _RACE_FULL_PARAMS or (param_set == _RACE_DYNAMIC_PARAMS and 'sig_i' in fixed_params):
        return 'race', fixed_params
    if param_set == _DDM_EXACT_PARAMS:
        return 'ddm_exact', fixed_params
    if param_set == _DDM_APPROX_PARAMS:
        return 'ddm_approx', fixed_params
    if _is_logistic(param_names):
        return 'logistic', fixed_params
    if _is_mlp(param_names):
        return 'mlp', fixed_params
    return 'unknown', fixed_params


def _curve_predictions(params_window, param_names, c_grid, model_family, fixed_params=None):
    sampled = _sample_curve_states(params_window)
    psych_curves = []
    rt_curves = []

    for col in range(sampled.shape[1]):
        params_vec = sampled[:, col]
        psych = rt = None

        if model_family == 'race':
            psych, rt = _race_curves(params_vec, param_names, c_grid, fixed_params=fixed_params)
        elif model_family == 'ddm_exact':
            psych, rt = _ddm_exact_curves(params_vec, param_names, c_grid)
        elif model_family == 'ddm_approx':
            psych, rt = _ddm_approx_curves(params_vec, param_names, c_grid)
        elif model_family == 'logistic':
            # For multi-input logistic, the psychometric x-axis is the
            # contrast input. Identify its weight (or fall back to the
            # first weight if no `c` channel is present); other inputs
            # are treated as 0 for the analytic curve.
            ib = param_names.index('b')
            if 'w' in param_names:
                iw = param_names.index('w')
            elif 'w_c' in param_names:
                iw = param_names.index('w_c')
            else:
                iw = 0  # first weight column
            psych = 1.0 / (1.0 + np.exp(-(params_vec[iw] * c_grid + params_vec[ib])))
        elif model_family == 'mlp':
            psych = _mlp_psychometric(params_vec, param_names, c_grid)

        if psych is not None:
            psych_curves.append(np.asarray(psych, dtype=float))
        if rt is not None:
            rt_curves.append(np.asarray(rt, dtype=float))

    if not psych_curves:
        return None, None

    psych_mean = np.mean(np.stack(psych_curves, axis=0), axis=0)
    rt_mean = np.nanmean(np.stack(rt_curves, axis=0), axis=0) if rt_curves else None
    return psych_mean, rt_mean


def _shared_ylim(series_list, pad_frac=0.05, min_pad=0.05):
    """Return a shared y-axis range covering all finite values in the series list."""
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


_TRAJ_COLORS = ['#4e9af1', '#f1a44e', '#4ef17a', '#f14e7a', '#c44ef1', '#f1f14e']


def _dopamine_tanh_readout(linear_pred, da_beta, da_offset):
    """Tanh dopamine readout used by the race model."""
    return np.tanh(0.5 * da_beta * (linear_pred - da_offset))


def _render_parameter_trajectories(result, *, key_suffix=''):
    """Inline plot of the K trial-by-trial parameter trajectories.

    Mirrors the corresponding block in Visualise Results so the Fit Model
    summary and the dedicated visualisation page render identical figures.

    Args:
        result      : dict — output of ``psytrax.fit`` (or the saved npy).
        key_suffix  : appended to widget keys so this helper can be called
                      multiple times on the same page without colliding
                      with itself.
    """
    params      = result['params']
    param_names = list(result['param_names'])
    K, N        = params.shape
    W_std       = (result.get('hess_info') or {}).get('W_std')
    dat         = result.get('data') or {}

    day_lengths = dat.get('dayLength') if dat.get('dayLength') is not None else np.array([])
    boundaries  = (
        np.cumsum(day_lengths).astype(int) if len(day_lengths) else np.array([], dtype=int)
    )
    trials = np.arange(N)

    traj_mode = st.radio(
        'Display mode',
        ['Separate', 'Combined'],
        horizontal=True,
        label_visibility='collapsed',
        key=f'fit_traj_mode{key_suffix}',
    )

    # Race-family layout: pair the symmetric weights left/right, then centre
    # the lapse-like z trajectory underneath at the same panel width.
    #
    #   wl | wr
    #   bl | br
    #      z
    #
    is_race = set(param_names) >= {'wr', 'wl', 'br', 'bl', 'z'}

    if traj_mode == 'Separate' and is_race:
        idx = {n: i for i, n in enumerate(param_names)}
        fig = plt.figure(figsize=(12, 7.6))
        _tc = _style_fig(fig)
        gs = fig.add_gridspec(3, 4, hspace=0.38, wspace=0.16)
        fig.subplots_adjust(left=0.06, right=0.985, top=0.96, bottom=0.075)
        ax_wl = fig.add_subplot(gs[0, 0:2])
        ax_wr = fig.add_subplot(gs[0, 2:4], sharey=ax_wl)
        ax_bl = fig.add_subplot(gs[1, 0:2])
        ax_br = fig.add_subplot(gs[1, 2:4], sharey=ax_bl)
        ax_z  = fig.add_subplot(gs[2, 1:3])

        def _draw(ax, name, idx_in_palette, hide_y=False):
            col = _TRAJ_COLORS[idx_in_palette % len(_TRAJ_COLORS)]
            _style_ax(ax, xlabel='Trial', title=name, title_pad=5)
            i = idx[name]
            ax.plot(trials, params[i], color=col, lw=0.8, alpha=0.9)
            if W_std is not None:
                ax.fill_between(trials,
                                params[i] - W_std[i], params[i] + W_std[i],
                                color=col, alpha=0.2)
            for b in boundaries[:-1]:
                ax.axvline(b, color=_tc['text'], lw=0.5, alpha=0.3, ls='--')
            if hide_y:
                # Right-column panels share y with the left; drop their
                # tick labels so the shared axis reads cleanly.
                plt.setp(ax.get_yticklabels(), visible=False)

        _draw(ax_wl, 'wl', 1)
        _draw(ax_wr, 'wr', 0, hide_y=True)
        _draw(ax_bl, 'bl', 3)
        _draw(ax_br, 'br', 2, hide_y=True)
        _draw(ax_z,  'z',  4)
        _show_fig(fig, 'param_trajectories.png')

    elif traj_mode == 'Separate':
        n_cols = min(K, 3)
        n_rows = int(np.ceil(K / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 3.6 * n_rows),
                                 squeeze=False,
                                 constrained_layout=True)
        _tc = _style_fig(fig)
        for k, (ax, name) in enumerate(zip(axes.flat, param_names)):
            col = _TRAJ_COLORS[k % len(_TRAJ_COLORS)]
            _style_ax(ax, xlabel='Trial', title=name)
            ax.plot(trials, params[k], color=col, lw=0.8, alpha=0.9)
            if W_std is not None:
                ax.fill_between(trials,
                                params[k] - W_std[k], params[k] + W_std[k],
                                color=col, alpha=0.2)
            for b in boundaries[:-1]:
                ax.axvline(b, color=_tc['text'], lw=0.5, alpha=0.3, ls='--')
        for ax in axes.flat[K:]:
            ax.set_visible(False)
        _show_fig(fig, 'param_trajectories.png')
    else:
        fig, ax = plt.subplots(figsize=(12, 4))
        _tc = _style_fig(fig)
        _style_ax(ax, xlabel='Trial', ylabel='Parameter value')
        for k, name in enumerate(param_names):
            col = _TRAJ_COLORS[k % len(_TRAJ_COLORS)]
            ax.plot(trials, params[k], color=col, lw=0.9, alpha=0.9, label=name)
            if W_std is not None:
                ax.fill_between(trials,
                                params[k] - W_std[k], params[k] + W_std[k],
                                color=col, alpha=0.15)
        for b in boundaries[:-1]:
            ax.axvline(b, color=_tc['text'], lw=0.5, alpha=0.3, ls='--')
        _style_legend(ax)
        fig.tight_layout(pad=1.3)
        _show_fig(fig, 'param_trajectories.png')


def _render_quartile_plots(result, *, n_win=4):
    """Inline psychometric / chronometric / dopamine quartile plots.

    Shared by the Fit Model post-fit summary and (eventually) Visualise
    Results.  Reads everything from ``result['data']`` plus the recovered
    ``params`` and ``model_hyper``.  Silently skips panels that don't apply
    (e.g. no RT for logistic fits, no dopamine for non-race fits).
    """
    params      = result['params']
    param_names = list(result['param_names'])
    K, N        = params.shape
    dat         = result.get('data') or {}

    if 'inputs' not in dat or 'c' not in dat['inputs'] or 'r' not in dat:
        st.info('Inline quartile plots need `c` and `r` in the fit data.')
        return

    c_data = np.asarray(dat['inputs']['c'])
    r_data = np.asarray(dat['r'])
    model_family, fixed_params = _model_family_info(param_names, result=result)

    _rt_key = next((k for k in ('T', 'times') if k in dat and dat[k] is not None), None)
    has_rt  = (_rt_key is not None) and model_family in _RT_CURVE_FAMILIES

    contrasts_unique = np.unique(c_data)
    c_grid = np.linspace(contrasts_unique.min(), contrasts_unique.max(), 100)
    edges = np.linspace(0, N, n_win + 1, dtype=int)

    # --- Psychometric quartiles ---------------------------------------
    st.subheader('Psychometric curve: evolution over learning')
    fig_p, axes_p = plt.subplots(2, 2, figsize=(11, 8))
    _tc = _style_fig(fig_p)
    for wi, ax in enumerate(axes_p.flat):
        t0, t1 = int(edges[wi]), int(edges[wi + 1])
        c_win  = c_data[t0:t1]
        r_win  = r_data[t0:t1]
        c_uniq = np.unique(c_win)
        p_emp  = np.array([r_win[c_win == cv].mean() for cv in c_uniq])
        n_w    = np.array([np.sum(c_win == cv) for cv in c_uniq])
        _style_ax(ax, xlabel='Signed contrast', ylabel='P(right)',
                  title=f'Trials {t0 + 1}–{t1}')
        ax.scatter(c_uniq, p_emp, s=[max(10, n / 5) for n in n_w],
                   color=_tc['text'], zorder=3)
        p_m, _ = _curve_predictions(
            params[:, t0:t1], param_names, c_grid, model_family,
            fixed_params=fixed_params,
        )
        if p_m is not None:
            ax.plot(c_grid, p_m, color='#4e9af1', lw=2)
        ax.axhline(0.5, color=_tc['text'], lw=0.5, ls='--', alpha=0.4)
        ax.axvline(0,   color=_tc['text'], lw=0.5, ls='--', alpha=0.4)
        ax.set_ylim(0, 1)
    fig_p.suptitle('Psychometric curve evolution',
                   color=_tc['text'], fontsize=13)
    fig_p.tight_layout()
    _show_fig(fig_p, 'psychometric_evolution.png')

    # --- Chronometric quartiles ---------------------------------------
    if has_rt:
        T_data = np.asarray(dat[_rt_key])
        st.subheader('Chronometric curve: evolution over learning')
        fig_c, axes_c = plt.subplots(2, 2, figsize=(11, 8))
        _tc = _style_fig(fig_c)
        panel_data = []
        y_series = []
        for wi in range(n_win):
            t0, t1 = int(edges[wi]), int(edges[wi + 1])
            c_win  = c_data[t0:t1]
            T_win  = T_data[t0:t1]
            c_uniq = np.unique(c_win)
            rt_emp = np.array([T_win[c_win == cv].mean() for cv in c_uniq])
            n_w    = np.array([np.sum(c_win == cv) for cv in c_uniq])
            _, rt_m = _curve_predictions(
                params[:, t0:t1], param_names, c_grid, model_family,
                fixed_params=fixed_params,
            )
            panel_data.append((t0, t1, c_uniq, rt_emp, n_w, rt_m))
            y_series.extend([rt_emp, rt_m])
        shared_ylim = _shared_ylim(y_series)
        for ax, (t0, t1, c_uniq, rt_emp, n_w, rt_m) in zip(
                axes_c.flat, panel_data):
            _style_ax(ax, xlabel='Signed contrast', ylabel='Mean RT (s)',
                      title=f'Trials {t0 + 1}–{t1}')
            ax.scatter(c_uniq, rt_emp, s=[max(10, n / 5) for n in n_w],
                       color=_tc['text'], zorder=3)
            if rt_m is not None:
                ax.plot(c_grid, rt_m, color='#4e9af1', lw=2)
            ax.axvline(0, color=_tc['text'], lw=0.5, ls='--', alpha=0.4)
            if shared_ylim is not None:
                ax.set_ylim(*shared_ylim)
        fig_c.suptitle('Chronometric curve evolution',
                       color=_tc['text'], fontsize=13)
        fig_c.tight_layout()
        _show_fig(fig_c, 'chronometric_evolution.png')

    # --- Dopamine quartiles (race + dopamine field) -------------------
    da_data = dat.get('dopamine')
    if model_family != 'race' or da_data is None:
        return
    da = np.asarray(da_data, dtype=float)
    if da.shape[0] != N:
        return
    try:
        wr_idx = param_names.index('wr')
        wl_idx = param_names.index('wl')
    except ValueError:
        return

    _model_mh = result.get('model_hyper') or {}
    _DA_BETA   = float(_model_mh.get('da_beta',   _DA_BETA_DEFAULT))
    _DA_OFFSET = float(_model_mh.get('da_offset', _DA_OFFSET_DEFAULT))

    st.subheader('Dopamine: empirical vs predicted, by quartile')
    fig_d, axes_d = plt.subplots(2, 2, figsize=(11, 8))
    _tc = _style_fig(fig_d)
    panel_data = []
    y_series = []
    for wi in range(n_win):
        t0, t1 = int(edges[wi]), int(edges[wi + 1])
        c_win  = c_data[t0:t1]
        da_win = da[t0:t1]
        finite = np.isfinite(da_win)
        c_f, da_f = c_win[finite], da_win[finite]
        c_uniq = np.unique(c_f) if c_f.size else np.array([])
        da_mean = np.array([
            da_f[c_f == cv].mean() for cv in c_uniq
        ]) if c_uniq.size else np.array([])
        da_sem = np.array([
            (da_f[c_f == cv].std()
             / max(np.sqrt(np.sum(c_f == cv)), 1.0))
            for cv in c_uniq
        ]) if c_uniq.size else np.array([])
        n_w = np.array([np.sum(c_f == cv) for cv in c_uniq])
        wr_w = params[wr_idx, t0:t1]
        wl_w = params[wl_idx, t0:t1]
        da_curve = np.array([
            np.mean(_dopamine_tanh_readout(
                np.where(cv >= 0, wr_w, wl_w) * abs(cv),
                _DA_BETA,
                _DA_OFFSET,
            ))
            for cv in c_grid
        ])
        panel_data.append((t0, t1, c_uniq, da_mean, da_sem, n_w, da_curve))
        y_series.extend([da_mean, da_curve])
    shared_ylim = _shared_ylim(y_series)
    for ax, (t0, t1, c_uniq, da_mean, da_sem, n_w, da_curve) in zip(
            axes_d.flat, panel_data):
        _style_ax(ax, xlabel='Signed contrast',
                  ylabel='Dopamine peak (a.u.)',
                  title=f'Trials {t0 + 1}–{t1}')
        if c_uniq.size:
            ax.errorbar(c_uniq, da_mean, yerr=da_sem,
                        fmt='o', color=_tc['text'], ecolor=_tc['text'],
                        elinewidth=0.8, capsize=2, markersize=5,
                        zorder=3)
        ax.plot(c_grid, da_curve, color='#4e9af1', lw=2)
        ax.axvline(0, color=_tc['text'], lw=0.5, ls='--', alpha=0.4)
        if shared_ylim is not None:
            ax.set_ylim(*shared_ylim)
    title_bits = []
    if 'sig_DA' in _model_mh:
        title_bits.append(f'σ_DA = {_model_mh["sig_DA"]:.3f}')
    title_bits.append(f'β = {_DA_BETA:.2f}')
    title_bits.append(f'centre = {_DA_OFFSET:.3f}')
    fig_d.suptitle('Dopamine: empirical vs predicted   '
                   '(' + ', '.join(title_bits) + ')',
                   color=_tc['text'], fontsize=13)
    fig_d.tight_layout()
    _show_fig(fig_d, 'dopamine_evolution.png')


st.set_page_config(page_title='psytrax', layout='wide')

_APP_DIR = os.path.dirname(__file__)
_DOC_ASSET_DIR = os.path.join(_APP_DIR, 'examples', 'assets')
_DAP014_TRAJ = os.path.join(_DOC_ASSET_DIR, 'dap014_race_trajectories.png')
_DAP014_PSY = os.path.join(_DOC_ASSET_DIR, 'dap014_race_psychometric.png')
_DAP014_CHRONO = os.path.join(_DOC_ASSET_DIR, 'dap014_race_chronometric.png')

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title('psytrax')
st.sidebar.caption('Empirical Bayes for trial-by-trial decision models')
page = st.sidebar.radio('Navigation', ['Instructions', 'Fit Model',
                                       'Visualise Results', 'Compare Models',
                                       'Model Recovery', 'IBL Explorer'],
                        label_visibility='collapsed')

# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------
if page == 'Instructions':
    st.title('psytrax')
    st.markdown('### Empirical Bayes fitting for trial-by-trial decision models')
    st.divider()

    st.markdown("""
psytrax fits a **Gaussian random-walk prior** over a sequence of K parameters
across N trials and optimises the prior variance (hyperparameters) by maximising
the marginal likelihood (evidence) using the Laplace approximation.

It is model-agnostic: you supply a **per-trial log-likelihood function** and psytrax
handles all the inference machinery.
""")

    st.subheader('See a real fit')
    st.caption('Example output from the bundled DAP014 race-model fit.')
    if all(os.path.exists(path) for path in (_DAP014_TRAJ, _DAP014_PSY, _DAP014_CHRONO)):
        st.image(_DAP014_TRAJ, caption='Parameter trajectories recovered across learning',
                 use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.image(_DAP014_PSY, caption='Psychometric evolution over learning',
                     use_container_width=True)
        with col2:
            st.image(_DAP014_CHRONO, caption='Chronometric evolution over learning',
                     use_container_width=True)
    else:
        st.info('Documentation example figures are missing from this checkout.')

    st.subheader('Quick start')
    st.code("""
import psytrax
from psytrax.models.race import log_lik_trial, N_PARAMS, PARAM_NAMES, default_hyper, default_E0

data = {
    'inputs':   {'c': contrast_array},  # signed contrast, shape (N,)
    'responses': response_array,        # 0 / 1, shape (N,)
    'times':     rt_array,              # reaction times, shape (N,)
    'session_lengths': day_length_array # trials per session, shape (n_sessions,)
}

result = psytrax.fit(
    data            = data,
    log_lik_trial   = log_lik_trial,
    n_params        = N_PARAMS,
    param_names     = PARAM_NAMES,
    hyper           = default_hyper(),
    E0              = default_E0(N),
    session_boundaries = True,
)

print(result['params'].shape)        # (K, N)
print(result['log_evidence'])        # scalar
""", language='python')

    st.subheader('Hyperparameters and what gets optimised')
    st.markdown("""
psytrax has three hyperparameters, all of which live in the `hyper` dict:

| Hyperparameter | Shape | Description | Optimised? |
|---|---|---|---|
| `sigma` | scalar or `(K,)` | Per-trial (within-session) process noise | **Always** |
| `sigInit` | `(K,)` | Initial uncertainty at trial 0 | **Never** — fixed prior |
| `sigDay` | scalar or `(K,)` | Extra process noise applied at session boundaries | Only when `session_boundaries=True` |
| `alpha` | scalar or `(K,)` | Learning rate scaling the learning rule output | Only when `learning_rule` is set |

By default only `sigma` is optimised.  To also optimise a larger jump at each session
boundary, pass `session_boundaries=True`:

```python
result = psytrax.fit(..., session_boundaries=True)
```

psytrax will initialise `sigDay` automatically if it is not already in `hyper`.
You can also supply it yourself:

```python
hyper = default_hyper()
hyper['sigDay'] = np.full(N_PARAMS, 2**-2)   # one value per parameter
result = psytrax.fit(..., hyper=hyper, session_boundaries=True)
```

To use a single scalar process noise shared across all K parameters (rather than
one per parameter), pass `shared_sigma=True` or build the hyper dict accordingly:

```python
result = psytrax.fit(..., shared_sigma=True)
# or
hyper = default_hyper(shared_sigma=True)
```

When a **learning rule** is supplied, psytrax also optimises per-parameter learning
rates `alpha` (scalar or `(K,)`) that scale the learning rule output.
""")

    st.subheader('Learning rules')
    st.markdown(r"""
By default the parameter transition is a zero-mean random walk:
$w_{t+1} - w_t \sim \mathcal{N}(0,\, \text{diag}(\sigma^2))$.
You can supply a **learning rule** so the transition has a non-zero mean:

$$w_{t+1} - w_t \sim \mathcal{N}\!\big(\text{diag}(\alpha)\,\hat{v}_t,\;\text{diag}(\sigma^2)\big)$$

where $\hat{v}_t = \text{learning\_rule}(w_t, \text{data}_t)$ is the raw update direction
and $\alpha$ is a vector of per-parameter learning rates optimised alongside $\sigma$
in the Empirical Bayes outer loop
([Ashwood, Roy et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/3a2f55e26e324b2c406d8b7df4607036-Abstract.html)).

**Built-in REINFORCE rule** — each built-in model provides `default_learning_rule()`
which implements the REINFORCE policy-gradient update:
$\hat{v}_t = \nabla_\theta \log p(y_t \mid x_t, \theta)\cdot r_t$.
Your data must include `data['inputs']['reward']` (1 = rewarded, 0 = unrewarded).

```python
from psytrax.models.logistic import (
    log_lik_trial, N_PARAMS, PARAM_NAMES, default_learning_rule,
)

result = psytrax.fit(
    data           = data,      # must include inputs['reward']
    log_lik_trial  = log_lik_trial,
    n_params       = N_PARAMS,
    learning_rule  = default_learning_rule(),
)
print(result['hyper']['alpha'])   # optimised learning rates
```

**Custom learning rules** — any JAX-traceable function with signature
`learning_rule(params, dat_trial) -> (K,)` can be passed to `psytrax.fit()`.
See the factories in `psytrax.learning_rules` (`make_reinforce`, `make_reinforce_baseline`)
for convenience wrappers.
""")

    st.subheader('Writing your own model')
    st.markdown("""
Provide any JAX-compatible per-trial function:

```python
import jax.numpy as jnp

def my_log_lik_trial(params, dat_trial):
    \"\"\"
    params    : jnp array (K,)  — parameters for this trial
    dat_trial : dict — same keys as your dat dict but scalar-valued per trial
                (psytrax vmaps over trials automatically)
    \"\"\"
    w, b = params
    p = jax.nn.sigmoid(w * dat_trial['inputs']['x'] + b)
    return dat_trial['r'] * jnp.log(p) + (1 - dat_trial['r']) * jnp.log(1 - p)

result = psytrax.fit(data=data, log_lik_trial=my_log_lik_trial, n_params=2)
```

The function must be written with **`jax.numpy`** (not `numpy`) so that psytrax
can differentiate through it to get the gradient and Hessian needed for MAP
estimation and the Laplace approximation.
""")

    st.subheader('GPU support')
    st.markdown("""
psytrax automatically selects the fastest safe execution path for the detected
hardware. On Apple Silicon it uses an experimental **hybrid** path:
Metal float32 for MAP optimisation, then CPU float64 for Hessian / evidence.
NVIDIA CUDA supports float64 directly and will accelerate fitting:

| Platform | Command |
|----------|---------|
| NVIDIA CUDA 12 | `pip install jax[cuda12]` |
| NVIDIA CUDA 11 | `pip install jax[cuda11_pip]` |

For most users, `device='auto'` is the recommended default.
""")

    st.subheader('Performance')
    st.markdown("""
Wall-clock fitting times on Apple M4 CPU, JAX L-BFGS optimizer, float64,
measured on bundled dopamine example mice. Times include warm-start, MAP/EB
cycles, joint choice + RT + dopamine likelihood, session-boundary process
noise, and trajectory credible bands (`hess_calc='weights'`).

| Mouse | Trials | Time |
|---|---:|---:|
| DAP044 | 407 | 0.72 min (43s) |
| DAP048 | 2,580 | 1.33 min (80s) |
| DAP110 | 2,583 | 1.03 min (62s) |
| DAP039 | 3,162 | 2.65 min (159s) |
| DAP014 | 3,213 | 1.32 min (79s) |
| DAP027 | 3,693 | 1.75 min (105s) |
| DAP033 | 4,019 | 2.92 min (175s) |
| DAP013 | 4,413 | 1.78 min (107s) |
| DAP009 | 4,885 | 2.02 min (121s) |
| DAP023 | 4,911 | 1.63 min (98s) |
| DAP017 | 5,078 | 2.35 min (141s) |
| DAP011 | 5,413 | 2.69 min (162s) |
| DAP015 | 5,989 | 3.38 min (203s) |
| DAP051 | 6,170 | 1.97 min (118s) |
| DAP156 | 6,371 | 2.60 min (156s) |
| DAP022 | 6,548 | 3.77 min (226s) |
| DAP046 | 7,796 | 4.36 min (261s) |
| DAP050 | 8,173 | 3.56 min (214s) |
| DAP028 | 8,408 | 2.99 min (180s) |
| DAP024 | 9,602 | 1.87 min (112s) |
| DAP007 | 12,018 | 2.46 min (147s) |

Across all 21 bundled dopamine example mice, the median fit time was 2.35 min
and the range was 0.72-4.36 min. These timings use real datasets rather than
synthetic sweeps, so they are not perfectly monotonic in trial count. NVIDIA
CUDA (float64) is expected to give a further **3-8x speedup** for models with
K >= 3 via `jax.vmap` on-device computation.
""")

    st.subheader('Installation')
    st.markdown("""
Core package:

```bash
pip install -e .
```

Streamlit app and plotting dependencies:

```bash
pip install -e .[web,ibl]
streamlit run app.py
```
""")

    st.subheader('IBL Explorer')
    st.markdown("""
The **IBL Explorer** page pulls public behavioural sessions directly from the
International Brain Laboratory archive through the **ONE** API and converts
them into psytrax's standard data dict.

- Public Alyx connection: the page authenticates against the public
  `openalyx.internationalbrainlab.org` endpoint automatically.
- Subject discovery: type 2 or more characters to get subject-name matches,
  then choose a subject to list available sessions.
- Supported trial layouts: the loader accepts the assembled `trials` object,
  `_ibl_trials.table.pqt`, and the older per-field `_ibl_trials.*.npy` layout.
- Conversion conventions: signed contrast is `contrastRight - contrastLeft`
  so positive values are rightward; `choice == -1` is mapped to a rightward
  response and `choice == +1` to leftward.
- RT extraction: the page prefers `firstMovement_times` relative to
  `stimOn_times` or `goCue_times`, and only uses raw `response_times` when
  they already look like true per-trial RTs.

For a notebook walkthrough of the same integration outside the app, see
`examples/ibl_one_integration_walkthrough.ipynb`.
""")

    st.subheader('Data format')
    st.markdown("""
| Key | Alias | Type | Description |
|-----|-------|------|-------------|
| `inputs` | — | `dict` | Dict of input arrays, each `(N, ...)` |
| `responses` | `r` | `array (N,)` | Response variable — discrete (e.g. 0/1) or continuous |
| `times` | `T` | `array (N,)` | Reaction times *(optional)* |
| `session_lengths` | `dayLength` | `array` | Trials per session *(optional)* |
""")

    st.subheader('Result dict keys')
    st.markdown("""
| Key | Shape | Description |
|-----|-------|-------------|
| `params` | `(K, N)` | MAP parameter estimates per trial |
| `param_names` | `list[str]` | Parameter names |
| `hyper` | `dict` | Optimised hyperparameters |
| `log_evidence` | `float` | Log marginal likelihood |
| `hess_info` | `dict` | `W_std`: credible intervals `(K, N)` |
| `lr_hat` | `(K, N-1)` | Raw learning-rule outputs per trial *(only when `learning_rule` is set)* |
| `duration` | `timedelta` | Wall-clock fitting time |
""")

# ---------------------------------------------------------------------------
# Fit Model
# ---------------------------------------------------------------------------
elif page == 'Fit Model':
    import os
    st.title('Fit Model')
    st.markdown(
        'Upload a dataset, choose a model and hyperparameters, and run `psytrax.fit()` '
        'directly from the browser.  The result can be downloaded and loaded into the '
        '**Visualise Results** or **Compare Models** pages.'
    )
    st.divider()

    # --- Data upload ---
    st.subheader('1. Load data')

    data_source = st.radio(
        'Data source',
        ['Example data', 'Upload my own file'],
        horizontal=True,
        key='fit_data_source',
    )

    import pandas as pd

    if data_source == 'Example data':
        _data_dir = os.path.join(os.path.dirname(__file__), 'data')
        _available_paths = {}
        if os.path.isdir(_data_dir):
            for f in sorted(os.listdir(_data_dir)):
                if not f.endswith('_data.npy'):
                    continue
                _available_paths[f.replace('_data.npy', '')] = (
                    os.path.join(_data_dir, f)
                )

        if not _available_paths:
            st.error('No example data found in `data/`. Run '
                     '`examples/extract_dopamine_to_data_files.py` first.')
            st.stop()

        animal = st.selectbox('Select mouse',
                              list(_available_paths.keys()),
                              key='fit_animal')
        raw = np.load(_available_paths[animal], allow_pickle=True).item()
        if 'dopamine' in raw:
            st.caption(f'Loaded with dopamine signal '
                       f'({np.sum(np.isfinite(np.asarray(raw["dopamine"])))} '
                       f'finite trials).')

    else:
        st.markdown("""
Upload a **`.npy`** file (pre-built data dict) or a **`.csv`** file and map its
columns to the required fields.

| Field | Required | Description |
|-------|----------|-------------|
| `inputs` | **Yes** | One or more columns used as model inputs |
| `responses` | **Yes** | Response column — numeric (discrete or continuous) or text labels (auto-mapped to 0/1) |
| `times` | No | Reaction-time column (seconds) |
| `session_id` | No | Column whose value identifies the session — used to compute session lengths |
""")

        data_file = st.file_uploader('Data file (.npy or .csv)', type=['npy', 'csv'], key='fit_data')
        if data_file is None:
            st.info('Upload a `.npy` or `.csv` file to continue.')
            st.stop()

    # Determine data format: CSV DataFrame (column mapping deferred until model
    # selection) vs ready-to-use .npy dict.
    _csv_df = None

    if data_source == 'Upload my own file' and data_file.name.endswith('.csv'):
        _csv_df = pd.read_csv(data_file)
        raw = None  # will be built after model selection + column mapping
        st.dataframe(_csv_df.head(5), use_container_width=True)
        st.caption(f'{len(_csv_df)} rows × {len(_csv_df.columns)} columns.  '
                   'Column mapping will appear after you choose a model below.')
    elif data_source == 'Upload my own file':
        raw = np.load(data_file, allow_pickle=True).item()  # .npy upload
    # else: example data — raw already loaded at line ~540

    # Show a quick summary for .npy / example data
    if raw is not None:
        _r_key  = 'responses' if 'responses' in raw else ('r' if 'r' in raw else None)
        _N_data = len(raw[_r_key]) if _r_key else '?'
        _has_rt  = any(k in raw for k in ('times', 'T'))
        _has_ses = any(k in raw for k in ('session_lengths', 'dayLength'))
        st.success(
            f'Ready: **{_N_data}** trials — '
            f'inputs: `{list(raw.get("inputs", {}).keys())}` — '
            f'RT: {"yes" if _has_rt else "no"} — '
            f'sessions: {"yes" if _has_ses else "no"}'
        )

    # --- Dopamine joint-fit checkbox (locks the model to the race family) ---
    _data_has_dopamine = (
        isinstance(raw, dict)
        and raw.get('dopamine') is not None
        and np.any(np.isfinite(np.asarray(raw['dopamine'], dtype=float)))
    )
    _race_with_dopamine = st.checkbox(
        'Include dopamine signal in fit (locks model to Race)',
        value=bool(_data_has_dopamine),
        key='fit_data_with_dopamine',
        disabled=not _data_has_dopamine,
        help=(
            'Adds a Gaussian likelihood term '
            'N(tanh(0.5 · da_beta · (w_eff · |c| − da_offset)), sig_DA²) '
            'for the per-trial dopamine peak. `sig_DA`, `da_beta`, and '
            '`da_offset` are estimated jointly with `sig_i` by Empirical Bayes. '
            'Available only when the loaded dataset contains a `dopamine` '
            'field — use the `with_dopamine ::` files in the dropdown above.'
        ),
    )
    if _race_with_dopamine and not _data_has_dopamine:
        _race_with_dopamine = False
    if not _data_has_dopamine:
        st.caption(
            ':grey[Dopamine fit disabled — the loaded dataset does not '
            'contain a `dopamine` field.]'
        )

    st.divider()

    # --- Model selection ---
    st.subheader('2. Choose model')
    if _race_with_dopamine:
        st.info('Model locked to **Race** because dopamine fitting is enabled. '
                'Uncheck the dopamine box above to choose a different model.')
        _model_options = ['Race model (inverse-Gaussian)']
    else:
        _model_options = [
            'Race model (inverse-Gaussian)',
            'DDM — exact (Navarro & Fuss 2009)',
            'Logistic regression',
        ]
    model_choice = st.selectbox(
        'Built-in model',
        _model_options,
        key='fit_model',
    )

    if model_choice == 'Race model (inverse-Gaussian)':
        from psytrax.models import race as _race_module
        from psytrax.models.race import (
            log_lik_trial as _llt,
            N_PARAMS as _K,
            PARAM_NAMES as _pnames,
            default_hyper as _dhyper,
            DATA_SPEC as _data_spec,
        )
        _race_fixed_sig_i = False  # legacy flag, kept for downstream branches

        if _race_with_dopamine:
            st.markdown("""
**Race model + joint dopamine fit** — two independent inverse-Gaussian
accumulators racing to threshold (`wr, wl, br, bl, z` evolve under the
random-walk prior, `sig_i` is an EB scalar).  In addition the per-trial
dopamine peak is modelled as `N(tanh(0.5 · da_beta · (w_eff · |c| − da_offset)),
sig_DA²)` with `w_eff = wr` if `c ≥ 0` else `wl`.  EB jointly optimises
`sig_i, sig_DA, da_beta, da_offset` with the random-walk variances.
""")
        else:
            st.markdown("""
**Race model** — two independent inverse-Gaussian accumulators racing to threshold.
The 5 trial-varying parameters (`wr, wl, br, bl, z`) evolve under a Gaussian
random-walk prior; the within-trial accumulator noise `sig_i` is a static
model-level hyperparameter, estimated jointly with `sigma` by Empirical
Bayes (you'll see the recovered value in the result).
""")
    elif model_choice == 'DDM — exact (Navarro & Fuss 2009)':
        from psytrax.models.ddm import (
            log_lik_trial as _llt,
            N_PARAMS as _K,
            PARAM_NAMES as _pnames,
            default_hyper as _dhyper,
            DATA_SPEC as _data_spec,
        )
        st.markdown("""
**DDM (exact)** — Wiener process between two absorbing barriers, using the
Navarro & Fuss (2009) / Bogacz et al. (2006) hybrid series solution.
3 parameters: `w` (contrast weight), `b` (drift bias), `a` (boundary
separation). The starting point is fixed at `a/2` (unbiased); bias is
captured exclusively by `b`, which removes the well-known b/z degeneracy.
""")
        _race_fixed_sig_i = False
        _race_with_dopamine = False
    else:
        from psytrax.models.logistic import make_model as _logistic_make_model
        _logistic_keys_str = st.text_input(
            'Logistic input regressors (comma-separated)',
            value='c',
            key='fit_logistic_inputs',
            help='Each name must correspond to a column under `data["inputs"]` '
                 '(or a CSV column you map below). Each input gets its own '
                 'trial-varying weight, fit jointly under the random-walk prior.',
        )
        _logistic_keys = [k.strip() for k in _logistic_keys_str.split(',') if k.strip()] or ['c']
        (_llt, _, _K, _pnames, _dhyper, _, _, _data_spec) = _logistic_make_model(_logistic_keys)
        _weight_summary = ', '.join(f'`{k}`' for k in _logistic_keys)
        st.markdown(f"""
**Logistic regression** — `P(right) = σ(w · x + b)`. {_K} parameters per trial:
weights for {_weight_summary}, plus a bias `b`. Each weight evolves under the
Gaussian random-walk prior.
""")
        _race_fixed_sig_i = False
        _race_with_dopamine = False

    # ------------------------------------------------------------------
    # Learning rule selection & construction
    # ------------------------------------------------------------------
    st.markdown('**Learning rule** *(optional)*')
    st.caption(
        'Shift the random-walk transition mean with a learning rule. '
        'psytrax will optimise per-parameter learning rates (α) alongside σ.'
    )

    _lr_choice = st.selectbox(
        'Learning rule',
        ['None', 'REINFORCE (built-in)', 'Upload custom (.py)'],
        key='fit_lr_choice',
    )

    _learning_rule = None
    _lr_reward_col = None

    if _lr_choice == 'REINFORCE (built-in)':
        from psytrax.learning_rules import augment_data_spec, make_reinforce
        _lr_reward_col = st.text_input(
            'Reward input key',
            value='reward',
            key='fit_lr_reward_key',
            help='Name for the reward signal in `data["inputs"]`. '
                 'Typically 1 = rewarded, 0 = unrewarded.',
        ).strip() or 'reward'
        # Augment DATA_SPEC so the reward column appears in the column-mapping
        # UI alongside the model's own inputs.
        _data_spec = augment_data_spec(_data_spec, make_reinforce(
            _llt, reward_key=_lr_reward_col))
        _learning_rule = make_reinforce(_llt, reward_key=_lr_reward_col)
        if _race_with_dopamine:
            st.info(
                'REINFORCE update direction = '
                '`∇_θ log p(choice, RT, dopamine | θ)` × reward.  Because '
                'the dopamine likelihood term is part of `log_lik_trial` '
                'when dopamine fitting is enabled, the score function '
                'naturally includes its gradient — no extra wiring needed. '
                'Per-parameter learning rates (α) are still EB-optimised '
                'alongside σ, sig_DA, da_beta, da_offset, sig_i.'
            )

    elif _lr_choice == 'Upload custom (.py)':
        st.markdown("""
Upload a `.py` file that defines a `learning_rule(params, dat_trial)` function.
The function must be JAX-traceable and return a `(K,)` array — the unnormalised
update direction for each parameter.

```python
import jax, jax.numpy as jnp

def learning_rule(params, dat_trial):
    # Example: simple gradient-weighted reward
    score = jax.grad(my_log_lik)(params, dat_trial)
    return score * dat_trial['inputs']['reward']
```
""")
        _lr_file = st.file_uploader('Learning rule file (.py)', type=['py'], key='fit_lr_upload')
        if _lr_file is not None:
            import importlib.util, tempfile, sys
            _lr_src = _lr_file.read().decode('utf-8')
            st.code(_lr_src, language='python')
            try:
                with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as _tmp:
                    _tmp.write(_lr_src)
                    _tmp_path = _tmp.name
                _spec = importlib.util.spec_from_file_location('_user_lr', _tmp_path)
                _lr_mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_lr_mod)
                if not hasattr(_lr_mod, 'learning_rule'):
                    st.error('The uploaded file must define a `learning_rule(params, dat_trial)` function.')
                else:
                    _learning_rule = _lr_mod.learning_rule
                    st.success('Custom learning rule loaded successfully.')
            except Exception as _lr_err:
                st.error(f'Failed to load learning rule: {_lr_err}')

    # Show the model's data requirements
    _req_inputs = list(_data_spec.get('inputs', {}).keys())
    _needs_rt = 'rt' in _data_spec
    st.caption(
        f'**Requires:** inputs `{_req_inputs}`'
        + (f', reaction times' if _needs_rt else '')
        + ', responses'
    )

    st.divider()

    # ------------------------------------------------------------------
    # Map data columns (model-driven)
    # ------------------------------------------------------------------
    st.subheader('3. Map data to model')

    if _csv_df is not None:
        # --- CSV: interactive column mapping driven by DATA_SPEC ---
        df = _csv_df
        cols = ['— none —'] + list(df.columns)
        num_cols = ['— none —'] + [c for c in df.columns
                                    if pd.api.types.is_numeric_dtype(df[c])]

        st.markdown('Map your CSV columns to the fields this model needs:')

        # --- Required inputs from DATA_SPEC ---
        _mapped_inputs = {}
        spec_inputs = _data_spec.get('inputs', {})
        n_spec_inputs = len(spec_inputs)
        if n_spec_inputs > 0:
            inp_cols_ui = st.columns(min(n_spec_inputs, 3))
            for idx, (inp_key, inp_info) in enumerate(spec_inputs.items()):
                with inp_cols_ui[idx % len(inp_cols_ui)]:
                    # Try to auto-detect a matching column
                    _auto_idx = next(
                        (i for i, c in enumerate(num_cols)
                         if c.lower() == inp_key.lower()
                         or c.lower() in (inp_key.lower(), f'signed_{inp_key.lower()}')),
                        0
                    )
                    chosen = st.selectbox(
                        f'**`{inp_key}`** — {inp_info["description"]}',
                        num_cols,
                        index=_auto_idx,
                        key=f'csv_input_{inp_key}',
                    )
                    if chosen != '— none —':
                        _mapped_inputs[inp_key] = chosen

        # --- Response and RT ---
        map_c1, map_c2 = st.columns(2)
        with map_c1:
            resp_col = st.selectbox(
                '**Response** — ' + _data_spec.get('response', {}).get('description', 'Response variable'),
                cols,
                index=next((i for i, c in enumerate(cols)
                            if c.lower() in ('r', 'response', 'responses', 'choice')), 0),
                key='csv_resp',
            )
        with map_c2:
            if _needs_rt:
                rt_col = st.selectbox(
                    '**RT** — ' + _data_spec['rt']['description'],
                    num_cols,
                    index=next((i for i, c in enumerate(num_cols)
                                if 'time' in c.lower() or c.lower() == 't'), 0),
                    key='csv_rt',
                )
            else:
                rt_col = '— none —'

        # --- Session ID (always optional, not model-specific) ---
        sess_col = st.selectbox(
            'Session-ID column *(optional — used to detect session boundaries)*',
            cols,
            index=next((i for i, c in enumerate(cols)
                        if 'sess' in c.lower() or 'day' in c.lower()), 0),
            key='csv_sess',
        )

        # --- Validate required mappings ---
        _missing = [k for k in spec_inputs if spec_inputs[k].get('required') and k not in _mapped_inputs]
        if _missing:
            st.warning(f'Please map the required input(s): {", ".join(_missing)}')
            st.stop()
        if resp_col == '— none —':
            st.warning('Please select a response column.')
            st.stop()
        if _needs_rt and rt_col == '— none —':
            st.warning('This model requires a reaction-time column.')
            st.stop()

        # --- Build raw dict from column mapping ---
        # Handle text-label responses (auto-map to 0/1)
        resp_raw = df[resp_col]
        if pd.api.types.is_numeric_dtype(resp_raw):
            resp_arr = resp_raw.to_numpy(dtype=float)
        else:
            unique_vals = resp_raw.dropna().unique()
            if len(unique_vals) != 2:
                st.error(f'Response column has {len(unique_vals)} unique values; expected 2 for text labels.')
                st.stop()
            unique_sorted = sorted(unique_vals)
            st.info(f'Mapping responses: `{unique_sorted[0]}` → 0, `{unique_sorted[1]}` → 1')
            resp_arr = resp_raw.map({unique_sorted[0]: 0.0, unique_sorted[1]: 1.0}).to_numpy(dtype=float)

        # Drop rows with NaN in required columns
        keep_mask = np.isfinite(resp_arr)
        for ic in _mapped_inputs.values():
            if pd.api.types.is_numeric_dtype(df[ic]):
                keep_mask &= np.isfinite(df[ic].to_numpy(dtype=float))
        if rt_col != '— none —':
            keep_mask &= np.isfinite(df[rt_col].to_numpy(dtype=float))
        if keep_mask.sum() < len(df):
            st.warning(f'Dropped {len(df) - keep_mask.sum()} rows with NaN/non-finite values.')
        df_clean = df[keep_mask].reset_index(drop=True)
        resp_arr = resp_arr[keep_mask]

        raw = {
            'inputs': {inp_key: df_clean[csv_col].to_numpy(dtype=float)
                       for inp_key, csv_col in _mapped_inputs.items()},
            'responses': resp_arr,
        }
        if rt_col != '— none —':
            raw['times'] = df_clean[rt_col].to_numpy(dtype=float)
        if sess_col != '— none —':
            from itertools import groupby as _groupby
            sess_vals = df_clean[sess_col].to_numpy()
            raw['session_lengths'] = np.array(
                [sum(1 for _ in g) for _, g in _groupby(sess_vals)]
            )

    elif raw is not None:
        # --- .npy / example data: validate against DATA_SPEC ---
        _input_keys = list(raw.get('inputs', {}).keys())
        _missing_inputs = [
            k for k, info in _data_spec.get('inputs', {}).items()
            if info.get('required') and k not in _input_keys
        ]
        _r_key = 'responses' if 'responses' in raw else ('r' if 'r' in raw else None)
        _has_rt = any(k in raw for k in ('times', 'T'))

        if _missing_inputs:
            st.warning(
                f'Your data is missing the input(s) this model expects: **{", ".join(_missing_inputs)}**. '
                f'Available inputs: `{_input_keys}`.  '
                'You can remap below.'
            )
            # Offer remapping for missing inputs
            for miss_key in _missing_inputs:
                info = _data_spec['inputs'][miss_key]
                remap = st.selectbox(
                    f'Map **`{miss_key}`** ({info["description"]}) to:',
                    ['— none —'] + _input_keys,
                    key=f'remap_{miss_key}',
                )
                if remap != '— none —':
                    # Create an alias: copy the existing input under the required key
                    raw['inputs'][miss_key] = raw['inputs'][remap]
            # Re-check
            _still_missing = [
                k for k in _missing_inputs if k not in raw.get('inputs', {})
            ]
            if _still_missing:
                st.error(f'Still missing required input(s): {", ".join(_still_missing)}')
                st.stop()

        if _needs_rt and not _has_rt:
            st.error('This model requires reaction times, but none were found in the data (`times` or `T`).')
            st.stop()
        if _r_key is None:
            st.error('No response variable found in the data (`responses` or `r`).')
            st.stop()

        st.success('Data matches model requirements.')
    else:
        st.error('No data loaded.')
        st.stop()

    # Final summary
    _r_key  = 'responses' if 'responses' in raw else ('r' if 'r' in raw else None)
    _N_data = len(raw[_r_key]) if _r_key else '?'
    _has_rt  = any(k in raw for k in ('times', 'T'))
    _has_ses = any(k in raw for k in ('session_lengths', 'dayLength'))
    if _csv_df is not None:
        st.success(
            f'Ready: **{_N_data}** trials — '
            f'inputs: `{list(raw.get("inputs", {}).keys())}` — '
            f'RT: {"yes" if _has_rt else "no"} — '
            f'sessions: {"yes" if _has_ses else "no"}'
        )

    st.divider()

    # --- Fitting options ---
    st.subheader('4. Configure fitting')

    col_a, col_b = st.columns(2)
    with col_a:
        n_trials_opt = st.number_input(
            'Max trials (0 = all)', min_value=0, value=0, step=100, key='fit_ntrials'
        )
        n_trials_opt = int(n_trials_opt) if n_trials_opt > 0 else None

        session_boundaries = st.checkbox(
            'Session boundaries (fit sigDay)',
            value=_has_ses,
            key='fit_session_boundaries',
        )
        shared_sigma = st.checkbox('Shared sigma (scalar, not per-parameter)', value=False,
                                   key='fit_shared_sigma')

    with col_b:
        map_tol = st.select_slider(
            'MAP tolerance',
            options=[1e-3, 1e-4, 1e-5, 1e-6],
            value=1e-4,
            format_func=lambda x: f'{x:.0e}',
            key='fit_map_tol',
        )
        subject_name = st.text_input('Subject name (used for filename)', value='subject',
                                     key='fit_subject')
        hess_calc = st.selectbox(
            'Credible intervals',
            ['All', 'weights', 'hyper', 'None'],
            index=0, key='fit_hess',
            help='`All` computes credible intervals for both the trial-varying '
                 'parameters and the optimised hyperparameters.  The hyper part '
                 'requires a small numerical Hessian (≈2n² extra likelihood '
                 'evaluations, where n is the number of hyperparameters) — '
                 'typically tens of seconds for a race fit.',
        )
        hess_calc = None if hess_calc == 'None' else hess_calc
        precision = 'float64'

    st.divider()

    # --- Sigma initialisation (expandable) ---
    with st.expander('Advanced: initial hyperparameters'):
        st.markdown(
            'Leave blank to use model defaults.  Values are in **log₂** scale '
            '(e.g. −3 → σ ≈ 0.125).'
        )
        default_h = _dhyper()
        sigma_init_str = st.text_input(
            f'sigma (log₂), {_K} values comma-separated or single scalar',
            value=', '.join(f'{np.log2(v):.1f}' for v in np.atleast_1d(default_h['sigma'])),
            key=f'fit_sigma_init_{model_choice}',
        )
        try:
            sigma_vals = [float(x.strip()) for x in sigma_init_str.split(',')]
            if len(sigma_vals) == 1:
                custom_sigma = float(2 ** sigma_vals[0])
            else:
                custom_sigma = 2 ** np.array(sigma_vals)
        except Exception:
            st.warning('Could not parse sigma — using model default.')
            custom_sigma = default_h['sigma']

        # Dopamine model_hyper starting values (race only, dopamine on).
        _custom_da_init = {}
        if _race_with_dopamine:
            st.markdown('---')
            st.markdown('**Dopamine model_hyper starting values** *(linear scale)*. '
                        'EB still optimises all four; these just set the initial '
                        'point.')
            from psytrax.models.race import (
                DEFAULT_SIG_I    as _DEFAULT_SIG_I,
                DEFAULT_SIG_DA   as _DEFAULT_SIG_DA,
                DEFAULT_DA_BETA  as _DEF_BETA,
                DEFAULT_DA_OFFSET as _DEF_OFFSET,
            )
            _da_init_specs = [
                ('sig_i',     'within-trial accumulator noise',    _DEFAULT_SIG_I),
                ('sig_DA',    'Gaussian std on dopamine peak',     _DEFAULT_SIG_DA),
                ('da_beta',   'tanh inverse temperature',          _DEF_BETA),
                ('da_offset', 'tanh centre on weighted contrast',   _DEF_OFFSET),
            ]
            _da_cols = st.columns(len(_da_init_specs))
            for (key, descr, default_val), _col in zip(_da_init_specs, _da_cols):
                with _col:
                    _val = st.number_input(
                        f'`{key}` init',
                        value=float(default_val),
                        min_value=1e-6,
                        step=0.05,
                        format='%.4f',
                        key=f'fit_da_init_{key}',
                        help=f'Initial {descr}. Must be > 0 (EB enforces '
                             'positivity via log₂ reparameterisation).',
                    )
                    if _val > 0:
                        _custom_da_init[key] = float(_val)

    # Build hyper dict
    hyper = _dhyper()
    # Respect the shared_sigma checkbox: collapse to a scalar if checked,
    # or ensure per-parameter array if unchecked.  Without this, the checkbox
    # had no effect because fit() only reads shared_sigma when hyper is None.
    if shared_sigma:
        if np.isscalar(custom_sigma):
            hyper['sigma'] = float(custom_sigma)
        else:
            hyper['sigma'] = float(np.mean(custom_sigma))
    else:
        if np.isscalar(custom_sigma):
            hyper['sigma'] = np.full(_K, float(custom_sigma))
        else:
            hyper['sigma'] = np.asarray(custom_sigma)

    st.divider()

    # --- Run ---
    st.subheader('5. Fit')
    st.info(
        'Streamlit-hosted fits can be noticeably slower than running psytrax '
        'locally, especially for dopamine race fits. This page is mainly for '
        'checking the workflow and inspecting intermediate behaviour; use a '
        'local Python run for production batches.',
    )

    if 'fit_running' not in st.session_state:
        st.session_state['fit_running'] = False
    if 'fit_result_path' not in st.session_state:
        st.session_state['fit_result_path'] = None
    if 'fit_log' not in st.session_state:
        st.session_state['fit_log'] = []
    if 'fit_error' not in st.session_state:
        st.session_state['fit_error'] = None

    run_btn = st.button('Run fit', disabled=st.session_state['fit_running'], key='fit_run')

    if run_btn:
        import psytrax
        import psytrax._hyper_opt as _hyper_opt_mod
        import time

        st.session_state['fit_running'] = True
        st.session_state['fit_result_path'] = None
        st.session_state['fit_log'] = []
        st.session_state['fit_error'] = None

        _q = queue.Queue()

        # Minimal tqdm shim: forwards each cycle and MAP-iteration update to the queue.
        # _n      = outer cycle count (incremented by update())
        # _map_n  = MAP iterations within the current cycle (reset on update())
        class _QueueTqdm:
            def __init__(self, *args, **kwargs):
                self._n     = 0
                self._map_n = 0
                self._postfix = {}
            def update(self, n=1):
                self._n    += n
                self._map_n = 0   # reset inner counter for the new cycle
                self._postfix.pop('MAP loss', None)
                _q.put(('progress', self._n, self._map_n, dict(self._postfix)))
            def set_postfix(self, d, **kwargs):
                self._postfix.update(d)
                if 'MAP loss' in d:
                    self._map_n += 1
                _q.put(('progress', self._n, self._map_n, dict(self._postfix)))
            def close(self): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass

        def _status_cb(payload):
            _q.put(('status', payload))

        _orig_tqdm = _hyper_opt_mod.tqdm
        _hyper_opt_mod.tqdm = _QueueTqdm

        def _run_fit():
            try:
                os.makedirs('fits', exist_ok=True)
                fit_kwargs = dict(
                    data=raw,
                    shared_sigma=shared_sigma,
                    session_boundaries=session_boundaries,
                    n_trials=n_trials_opt,
                    hess_calc=hess_calc,
                    map_tol=float(map_tol),
                    precision=precision,
                    subject_name=subject_name,
                    save=True,
                    verbose=True,
                    status_callback=_status_cb,
                )

                fit_kwargs.update(
                    log_lik_trial=_llt,
                    n_params=_K,
                    param_names=_pnames,
                )

                # Defensive: force-enable model_hyper optimisation explicitly
                # so even an older deployed psytrax (where the default may
                # differ) still EB-optimises every model_hyper key.
                fit_kwargs['optimise_model_hyper'] = True

                # Joint dopamine fit (race only, opt-in via UI checkbox).
                if _race_with_dopamine:
                    _mh_init = _race_module.default_model_hyper_with_dopamine()
                    # Defensive: older deployed psytrax versions returned
                    # only {sig_i, sig_DA} from default_model_hyper_with_dopamine.
                    # Force-include da_beta / da_offset so EB optimises them
                    # regardless of which version is in site-packages.
                    _mh_init.setdefault(
                        'da_beta',
                        float(getattr(_race_module, 'DEFAULT_DA_BETA', 2.0)),
                    )
                    _mh_init.setdefault(
                        'da_offset',
                        float(getattr(_race_module, 'DEFAULT_DA_OFFSET', 0.001)),
                    )
                    # Apply user-defined starting values from the advanced
                    # expander (any subset; missing keys keep the model
                    # default).
                    for k, v in (_custom_da_init or {}).items():
                        if k in _mh_init and v is not None:
                            _mh_init[k] = float(v)
                    fit_kwargs['model_hyper'] = _mh_init
                    _status_cb({
                        'stage':   'setup',
                        'message': (
                            'Dopamine term enabled — sig_DA, da_beta and '
                            'da_offset will be EB-optimised.  Initial '
                            f'model_hyper = {_mh_init}.'
                        ),
                    })

                # Pass learning rule if one was selected
                if _learning_rule is not None:
                    fit_kwargs['learning_rule'] = _learning_rule
                    _status_cb({'stage': 'setup', 'message': 'Learning rule enabled — alpha will be optimised.'})

                result = psytrax.fit(
                    hyper=hyper,
                    **fit_kwargs,
                )
                _q.put(('done', result))
            except Exception as e:
                import traceback
                _q.put(('error', traceback.format_exc()))
            finally:
                _hyper_opt_mod.tqdm = _orig_tqdm

        _thread = threading.Thread(target=_run_fit, daemon=True)
        _thread.start()
        st.session_state['_fit_thread'] = _thread
        st.session_state['_fit_queue'] = _q

    if st.session_state['fit_running']:
        import time
        _q      = st.session_state['_fit_queue']
        _thread = st.session_state['_fit_thread']

        st.markdown('**Fitting in progress…** &nbsp; `JAX L-BFGS`')
        col_cyc, col_map = st.columns(2)
        cycle_text   = col_cyc.empty()
        map_text     = col_map.empty()
        status_text  = st.empty()
        log_evd_text = st.empty()
        log_box      = st.empty()

        cycle, map_iter, log_evd_str, best_str, map_loss_str = 0, 0, '—', '—', '—'
        current_status = 'Preparing fit…'
        fit_log = st.session_state.get('fit_log', [])
        terminal = None

        while _thread.is_alive():
            while not _q.empty():
                try:
                    msg = _q.get_nowait()
                    if msg[0] == 'progress':
                        _, cycle, map_iter, postfix = msg
                        log_evd_str = postfix.get('log_evd', '—')
                        best_str = postfix.get('best', '—')
                        map_loss_str = postfix.get('MAP loss', map_loss_str)
                    elif msg[0] == 'status':
                        payload = msg[1]
                        current_status = payload.get('message', current_status)
                        fit_log.append(current_status)
                        fit_log = fit_log[-12:]
                        st.session_state['fit_log'] = fit_log
                    elif msg[0] in ('done', 'error'):
                        terminal = msg
                except queue.Empty:
                    break
            cycle_text.metric('Cycles completed', cycle)
            map_text.metric('MAP iters (current cycle)', map_iter)
            status_text.markdown(f'**Current step:** {current_status}')
            log_evd_text.markdown(
                f'Log evidence (higher is better) — current: **{log_evd_str}** '
                f'&nbsp;|&nbsp; best: **{best_str}**'
                + (
                    f' &nbsp;|&nbsp; Neg. log posterior (lower is better): **{map_loss_str}**'
                    if map_loss_str != '—' else ''
                )
            )
            if fit_log:
                log_box.code('\n'.join(fit_log), language='text')
            time.sleep(0.5)

        while not _q.empty():
            try:
                msg = _q.get_nowait()
                if msg[0] == 'progress':
                    _, cycle, map_iter, postfix = msg
                    log_evd_str = postfix.get('log_evd', log_evd_str)
                    best_str = postfix.get('best', best_str)
                    map_loss_str = postfix.get('MAP loss', map_loss_str)
                elif msg[0] == 'status':
                    payload = msg[1]
                    current_status = payload.get('message', current_status)
                    fit_log.append(current_status)
                    fit_log = fit_log[-12:]
                elif msg[0] in ('done', 'error'):
                    terminal = msg
            except queue.Empty:
                break

        st.session_state['fit_log'] = fit_log
        st.session_state['fit_running'] = False
        if terminal is None:
            terminal = ('error', 'No result received from fitting thread.')
        msg_type, payload = terminal[0], terminal[1]
        if msg_type == 'done':
            st.session_state['fit_result_path'] = payload
        else:
            st.session_state['fit_error'] = payload

        st.rerun()

    if st.session_state['fit_error']:
        st.error(f'Fitting failed:\n\n```\n{st.session_state["fit_error"]}\n```')

    if st.session_state['fit_result_path']:
        path = st.session_state['fit_result_path']
        res = np.load(path, allow_pickle=True).item()
        st.success(f'Fit complete! Saved to `{path}`')
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Trials', res['params'].shape[1])
        c2.metric('Parameters', res['params'].shape[0])
        c3.metric('Log evidence', f"{res['log_evidence']:.1f}")
        c4.metric('Duration', str(res['duration']).split('.')[0])
        execution = res.get('execution')
        if execution:
            st.caption(
                "Execution: "
                f"{execution.get('description', 'unknown')} "
                f"(MAP {execution.get('map_precision', '?')}, "
                f"evidence {execution.get('evidence_precision', '?')})"
            )

        # Initial → recovered model_hyper table (so it's immediately
        # obvious whether each scalar moved during EB).
        _res_mh = res.get('model_hyper') or {}
        if _res_mh:
            import pandas as pd
            _init_mh = (
                fit_kwargs.get('model_hyper') if 'fit_kwargs' in dir() else None
            )
            # fit_kwargs is local to _run_fit; recover the initial values
            # we'd have used so the comparison still works.
            if _init_mh is None:
                if _race_with_dopamine:
                    _init_mh = _race_module.default_model_hyper_with_dopamine()
                    _init_mh.setdefault(
                        'da_beta',
                        float(getattr(_race_module, 'DEFAULT_DA_BETA', 2.0)),
                    )
                    _init_mh.setdefault(
                        'da_offset',
                        float(getattr(_race_module, 'DEFAULT_DA_OFFSET', 0.001)),
                    )
                    for k, v in (_custom_da_init or {}).items():
                        if k in _init_mh and v is not None:
                            _init_mh[k] = float(v)
                else:
                    _init_mh = {}
            mh_rows = []
            for k in sorted(_res_mh.keys()):
                init_v = _init_mh.get(k)
                rec_v  = float(_res_mh[k])
                moved  = (init_v is not None
                          and not np.isclose(init_v, rec_v, rtol=1e-3))
                mh_rows.append({
                    'hyperparameter': k,
                    'initial': '—' if init_v is None else f'{init_v:.4f}',
                    'recovered': f'{rec_v:.4f}',
                    'moved by EB': '✓' if moved else '·',
                })
            st.markdown('**Optimised model-level hyperparameters**')
            st.dataframe(pd.DataFrame(mh_rows).set_index('hyperparameter'),
                         use_container_width=True)

        with open(path, 'rb') as f:
            st.download_button(
                'Download fit file (.npy)',
                data=f.read(),
                file_name=os.path.basename(path),
                mime='application/octet-stream',
                key='fit_download',
            )

        # Inline summary plots: parameter trajectories, then psychometric,
        # chronometric (where applicable), and dopamine (when present in
        # data + race fit).
        st.divider()
        st.subheader('Inferred parameter trajectories')
        try:
            _render_parameter_trajectories(res, key_suffix='_fit_post')
        except Exception as exc:
            st.warning(f'Could not render trajectories: {exc}')

        st.divider()
        st.subheader('Behavioural & dopamine validation plots')
        st.caption(
            'Empirical bins (black) vs analytic curves from the recovered '
            'trajectory (blue), broken down into four equally-sized trial '
            'windows (early → late).  For the full breakdown '
            '(hyperparameter table, credible intervals, …) open the saved '
            'file in **Visualise Results**.'
        )
        try:
            _render_quartile_plots(res)
        except Exception as exc:
            st.warning(f'Could not render quartile plots: {exc}')

        st.info('Load this file in **Visualise Results** or **Compare Models** for the full breakdown.')

# ---------------------------------------------------------------------------
# IBL Explorer
# ---------------------------------------------------------------------------
elif page == 'IBL Explorer':
    import pandas as pd

    st.title('IBL Explorer')
    st.markdown(
        'Load behavioural data from the '
        '[International Brain Laboratory](https://www.internationalbrainlab.com/) '
        'public archive via the **ONE** API, fit a psytrax model, and visualise '
        'the results — all in one page.'
    )
    st.divider()

    # ==================================================================
    # 1. ONE Connection
    # ==================================================================
    st.subheader('1. Connect to IBL')
    _IBL_BASE_URL = 'https://openalyx.internationalbrainlab.org'
    _IBL_PUBLIC_USER = 'intbrainlab'
    _IBL_PUBLIC_PASSWORD = 'international'

    try:
        from one.api import ONE
        _ONE_AVAILABLE = True
    except ImportError:
        _ONE_AVAILABLE = False

    if not _ONE_AVAILABLE:
        st.error(
            '**`ONE-api` is not installed.**  \n'
            'Install the IBL explorer dependency with:\n\n'
            '```bash\n'
            'pip install ONE-api\n'
            '# or, from this repo:\n'
            'pip install -e .[web,ibl]\n'
            '```\n\n'
            'Then restart the app.'
        )
        st.stop()

    # Initialise session-state keys
    for _k, _v in [
        ('one_client',  None), ('one_connected', False),
        ('ibl_subjects', []),  ('ibl_eids', []),
        ('ibl_subject_selected', None),
        ('ibl_data',    None), ('ibl_fit_running', False),
        ('ibl_fit_result_path', None), ('ibl_fit_log', []),
        ('ibl_fit_error', None),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    if not st.session_state['one_connected']:
        st.markdown(
            'Connect to the **IBL public server** '
            '(`openalyx.internationalbrainlab.org`).  '
            'No credentials are required for public data.'
        )
        if st.button('Connect', key='ibl_connect'):
            with st.spinner('Connecting…'):
                try:
                    # ONE-api silently falls back to an incomplete saved public profile unless the
                    # public Alyx password is provided explicitly here.
                    _one = ONE(
                        base_url=_IBL_BASE_URL,
                        username=_IBL_PUBLIC_USER,
                        password=_IBL_PUBLIC_PASSWORD,
                        silent=True,
                    )
                    st.session_state['one_client'] = _one
                    st.session_state['one_connected'] = True
                    st.rerun()
                except Exception as _e:
                    st.error(f'Connection failed: {_e}')
        st.stop()

    st.success('Connected to IBL public server.')
    one = st.session_state['one_client']
    st.divider()

    # ==================================================================
    # 2. Subject browser
    # ==================================================================
    st.subheader('2. Select subject')

    @st.cache_data(show_spinner='Searching subjects…')
    def _search_subjects(query):
        query = query.strip()
        if len(query) < 2:
            return []

        _matches = one.alyx.rest(
            'subjects',
            'list',
            django=f'nickname__istartswith,{query}',
            limit=25,
        )
        _matches = _matches[:25] if not isinstance(_matches, list) else _matches
        if not _matches:
            _matches = one.alyx.rest(
                'subjects',
                'list',
                django=f'nickname__icontains,{query}',
                limit=25,
            )
            _matches = _matches[:25] if not isinstance(_matches, list) else _matches
        return list(dict.fromkeys(rec['nickname'] for rec in _matches))

    _subject_query = st.text_input(
        'Search subject name (e.g. `KS023`, `ZM_2241`, `CSHL049`)',
        key='ibl_subject_query',
    ).strip()

    if len(_subject_query) < 2:
        st.info('Type at least 2 characters to see matching subject names.')
        st.stop()

    try:
        _subject_matches = _search_subjects(_subject_query)
    except Exception as _e:
        st.error(f'Failed to search subjects: {_e}')
        st.stop()

    st.session_state['ibl_subjects'] = _subject_matches
    if st.session_state.get('ibl_subject_selected') not in _subject_matches:
        st.session_state['ibl_subject_selected'] = None

    if not _subject_matches:
        st.warning(f'No subjects matched **{_subject_query}**.')
        st.stop()

    _exact = next((s for s in _subject_matches if s.lower() == _subject_query.lower()), None)
    _subject = st.selectbox(
        'Matching subjects',
        _subject_matches,
        index=_subject_matches.index(_exact) if _exact else None,
        placeholder='Select a subject',
        key='ibl_subject_selected',
    )

    if not _subject:
        st.info('Select one of the matching subject names to search for sessions.')
        st.stop()

    st.divider()

    # ==================================================================
    # 3. Session selection
    # ==================================================================
    st.subheader('3. Select sessions')

    @st.cache_data(show_spinner='Searching sessions…')
    def _search_sessions(subject):
        """Return list of session summaries for an exact subject nickname."""
        eids, details = one.search(subject=subject, details=True)
        sessions = []
        for eid, det in zip(eids, details):
            if det.get('subject') != subject:
                continue
            date_str = str(det.get('date') or det.get('start_time', '')[:10] or eid)
            sessions.append({
                'eid': str(det.get('id', eid)),
                'date': date_str,
                'lab': det.get('lab', '?'),
                'number': det.get('number', '?'),
                'task_protocol': det.get('task_protocol', '?'),
            })
        # Sort by date
        sessions.sort(key=lambda s: s['date'])
        return sessions

    try:
        _sessions = _search_sessions(_subject)
    except Exception as _e:
        st.error(f'Failed to search sessions: {_e}')
        st.stop()

    if not _sessions:
        st.warning(f'No sessions found for subject **{_subject}**.')
        st.stop()

    st.caption(f'Found **{len(_sessions)}** session(s) for **{_subject}**.')

    # Date range filter
    _all_dates = [s['date'] for s in _sessions]
    _date_col1, _date_col2 = st.columns(2)
    with _date_col1:
        _date_from = st.selectbox('From date', _all_dates, index=0, key='ibl_date_from')
    with _date_col2:
        _date_to = st.selectbox('To date', _all_dates, index=len(_all_dates) - 1, key='ibl_date_to')

    _filtered = [s for s in _sessions if _date_from <= s['date'] <= _date_to]
    st.caption(f'**{len(_filtered)}** session(s) in selected range.')

    if not _filtered:
        st.warning('No sessions in the selected date range.')
        st.stop()

    # Show session table
    _sess_df = pd.DataFrame(_filtered)
    st.dataframe(
        _sess_df[['date', 'number', 'lab', 'task_protocol', 'eid']],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ==================================================================
    # 4. Load & preview data
    # ==================================================================
    st.subheader('4. Load data')

    def _load_ibl_trials(eid):
        """Load trials for one session, supporting both legacy arrays and table parquet."""
        _errors = []

        for _kwargs in (
            {},
            {'collection': 'alf'},
            {'namespace': 'ibl'},
            {'namespace': 'ibl', 'collection': 'alf'},
        ):
            try:
                return one.load_object(eid, 'trials', **_kwargs)
            except Exception as _e:
                _label = 'load_object' if not _kwargs else f'load_object{_kwargs}'
                _errors.append(f'{_label}: {_e}')

        try:
            _table = one.load_dataset(eid, '_ibl_trials.table.pqt')
            if isinstance(_table, pd.DataFrame):
                return _table
        except Exception as _e:
            _errors.append(f'load_dataset: {_e}')

        try:
            _table_path = one.load_dataset(eid, '_ibl_trials.table.pqt', download_only=True)
            return pd.read_parquet(_table_path)
        except Exception as _e:
            _errors.append(f'read_parquet: {_e}')

        _required = {
            'choice': '_ibl_trials.choice.npy',
            'contrastLeft': '_ibl_trials.contrastLeft.npy',
            'contrastRight': '_ibl_trials.contrastRight.npy',
            'response_times': '_ibl_trials.response_times.npy',
        }
        _optional = {
            'feedbackType': '_ibl_trials.feedbackType.npy',
            'rewardVolume': '_ibl_trials.rewardVolume.npy',
            'probabilityLeft': '_ibl_trials.probabilityLeft.npy',
            'stimOn_times': '_ibl_trials.stimOn_times.npy',
            'feedback_times': '_ibl_trials.feedback_times.npy',
            'goCue_times': '_ibl_trials.goCue_times.npy',
            'firstMovement_times': '_ibl_trials.firstMovement_times.npy',
            'intervals': '_ibl_trials.intervals.npy',
        }

        _trial_dict = {}
        _missing_required = []
        for _key, _dataset in _required.items():
            try:
                _trial_dict[_key] = one.load_dataset(eid, _dataset)
            except Exception as _e:
                _missing_required.append(_dataset)
                _errors.append(f'manual {_dataset}: {_e}')

        if not _missing_required:
            for _key, _dataset in _optional.items():
                try:
                    _trial_dict[_key] = one.load_dataset(eid, _dataset)
                except Exception:
                    pass
            return _trial_dict

        raise RuntimeError('; '.join(_errors))

    def _trial_array(trials, key):
        """Extract a trial field as a numpy array from either an AlfBunch or DataFrame."""
        if isinstance(trials, pd.DataFrame):
            if key in trials.columns:
                return np.asarray(trials[key], dtype=float)
            return None
        if hasattr(trials, key):
            return np.asarray(getattr(trials, key), dtype=float)
        if isinstance(trials, dict) and key in trials:
            return np.asarray(trials[key], dtype=float)
        return None

    def _relative_rt_candidates(n_trials, anchor, event):
        """Return per-trial event-anchor differences where both timestamps are valid."""
        if anchor is None or event is None:
            return np.full(n_trials, np.nan)
        _delta = np.asarray(event, dtype=float) - np.asarray(anchor, dtype=float)
        _valid = np.isfinite(_delta) & (_delta > 0)
        return np.where(_valid, _delta, np.nan)

    def _looks_like_relative_rt(raw_rt):
        """Heuristic to distinguish per-trial RTs from absolute session timestamps."""
        raw_rt = np.asarray(raw_rt, dtype=float)
        finite = raw_rt[np.isfinite(raw_rt)]
        if finite.size == 0:
            return False
        if np.nanpercentile(finite, 95) <= 20:
            return True
        if finite.size < 3:
            return False
        # Absolute timestamps are typically large and almost monotonically increasing.
        return np.mean(np.diff(finite) >= 0) < 0.95

    def _ibl_rt(trials):
        """Estimate per-trial RTs from the most informative available timing field."""
        response_times = _trial_array(trials, 'response_times')
        if response_times is None:
            raise KeyError('response_times')

        n_trials = len(response_times)
        stim_on = _trial_array(trials, 'stimOn_times')
        first_move = _trial_array(trials, 'firstMovement_times')
        go_cue = _trial_array(trials, 'goCue_times')

        rt = np.full(n_trials, np.nan)

        # Prefer movement onset relative to stimulus onset when available.
        for candidate in (
            _relative_rt_candidates(n_trials, stim_on, first_move),
            _relative_rt_candidates(n_trials, stim_on, response_times),
            _relative_rt_candidates(n_trials, go_cue, first_move),
            _relative_rt_candidates(n_trials, go_cue, response_times),
        ):
            use = np.isnan(rt) & np.isfinite(candidate)
            rt[use] = candidate[use]

        # Only fall back to the raw field when it already looks like a true RT array.
        if _looks_like_relative_rt(response_times):
            use = np.isnan(rt) & np.isfinite(response_times) & (response_times > 0)
            rt[use] = response_times[use]

        return rt

    def _ibl_to_psytrax(trials_list):
        """Convert list of IBL trial dicts/Bunches to a psytrax data dict."""
        all_c, all_r, all_t, all_reward = [], [], [], []
        all_p_left = []
        session_lengths = []

        for trials in trials_list:
            # ---- Signed contrast ----
            cL = _trial_array(trials, 'contrastLeft')
            cR = _trial_array(trials, 'contrastRight')
            if cL is None or cR is None:
                raise KeyError('contrastLeft/contrastRight')
            cL = np.nan_to_num(cL, nan=0.0)
            cR = np.nan_to_num(cR, nan=0.0)
            c = cR - cL  # positive = rightward stimulus

            # ---- Choice: in IBL, -1 = rightward, +1 = leftward ----
            choice = _trial_array(trials, 'choice')
            if choice is None:
                raise KeyError('choice')
            r = np.where(choice == -1.0, 1.0, np.where(choice == 1.0, 0.0, np.nan))

            # ---- Reward / feedback: IBL uses -1 (error) / +1 (correct) → 0 / 1 ----
            fb = _trial_array(trials, 'feedbackType')
            if fb is not None:
                reward = np.where(fb == -1.0, 0.0, np.where(fb == 1.0, 1.0, np.nan))
            else:
                reward_volume = _trial_array(trials, 'rewardVolume')
                if reward_volume is None:
                    raise KeyError('feedbackType/rewardVolume')
                reward = (reward_volume > 0).astype(float)

            # ---- Reaction time ----
            rt = _ibl_rt(trials)

            # ---- Optional extras ----
            p_left = _trial_array(trials, 'probabilityLeft')
            if p_left is None:
                p_left = np.full(len(c), 0.5)

            # ---- Filter invalid trials (NaN choice, etc.) ----
            valid = (
                np.isfinite(c) &
                np.isfinite(r) &
                np.isfinite(rt) &
                np.isfinite(reward) &
                np.isfinite(p_left) &
                (rt > 0)
            )
            c, r, rt, reward, p_left = c[valid], r[valid], rt[valid], reward[valid], p_left[valid]

            all_c.append(c)
            all_r.append(r)
            all_t.append(rt)
            all_reward.append(reward)
            all_p_left.append(p_left)
            session_lengths.append(len(c))

        return {
            'inputs': {
                'c': np.concatenate(all_c),
                'reward': np.concatenate(all_reward),
                'p_left': np.concatenate(all_p_left),
            },
            'responses': np.concatenate(all_r),
            'times': np.concatenate(all_t),
            'session_lengths': np.array(session_lengths, dtype=int),
        }

    _selected_eids = [s['eid'] for s in _filtered]

    if st.button('Load trials from selected sessions', key='ibl_load'):
        _trials_list = []
        _load_bar = st.progress(0, text='Loading…')
        for _i, _eid in enumerate(_selected_eids):
            _load_bar.progress((_i + 1) / len(_selected_eids),
                               text=f'Loading session {_i + 1}/{len(_selected_eids)}…')
            try:
                _tr = _load_ibl_trials(_eid)
                _trials_list.append(_tr)
            except Exception as _e:
                st.warning(f'Skipped session `{_eid}`: {_e}')
        _load_bar.empty()

        if not _trials_list:
            st.error('No trial data could be loaded.')
            st.stop()

        _raw = _ibl_to_psytrax(_trials_list)
        st.session_state['ibl_data'] = _raw
        # Clear any previous fit when new data is loaded
        st.session_state['ibl_fit_result_path'] = None
        st.session_state['ibl_fit_error'] = None
        st.rerun()

    raw = st.session_state.get('ibl_data')
    if raw is None:
        st.info('Click **Load trials** to fetch data from the IBL server.')
        st.stop()

    _N_data = len(raw['responses'])
    _n_sess = len(raw['session_lengths'])
    _has_rt = 'times' in raw
    _has_ses = 'session_lengths' in raw

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Trials', _N_data)
    c2.metric('Sessions', _n_sess)
    c3.metric('Has RT', 'yes' if _has_rt else 'no')
    c4.metric('Has reward', 'reward' in raw.get('inputs', {}))

    with st.expander('Preview data (first 200 trials)'):
        _n_preview = min(200, _N_data)
        st.dataframe(pd.DataFrame({
            'contrast': raw['inputs']['c'][:_n_preview],
            'response': raw['responses'][:_n_preview],
            'RT (s)': raw['times'][:_n_preview] if _has_rt else [None] * _n_preview,
            'reward': raw['inputs'].get('reward', np.full(_n_preview, np.nan))[:_n_preview],
            'p_left': raw['inputs'].get('p_left', np.full(_n_preview, np.nan))[:_n_preview],
        }), use_container_width=True, hide_index=True)

    st.divider()

    # ==================================================================
    # 5. Choose model & learning rule
    # ==================================================================
    st.subheader('5. Choose model')

    _ibl_model = st.selectbox(
        'Built-in model',
        ['Race model (inverse-Gaussian)', 'DDM — exact (Navarro & Fuss 2009)',
         'Logistic regression'],
        key='ibl_model',
    )

    if _ibl_model == 'Race model (inverse-Gaussian)':
        from psytrax.models.race import (
            log_lik_trial as _ibl_llt,
            N_PARAMS as _ibl_K,
            PARAM_NAMES as _ibl_pnames,
            default_hyper as _ibl_dhyper,
            DATA_SPEC as _ibl_data_spec,
        )
        _ibl_race_fixed = False  # legacy flag (sig_i is now an EB-estimated model_hyper)
    elif _ibl_model == 'DDM — exact (Navarro & Fuss 2009)':
        from psytrax.models.ddm import (
            log_lik_trial as _ibl_llt, N_PARAMS as _ibl_K,
            PARAM_NAMES as _ibl_pnames, default_hyper as _ibl_dhyper,
            DATA_SPEC as _ibl_data_spec,
        )
        _ibl_race_fixed = False
    else:
        from psytrax.models.logistic import make_model as _ibl_logistic_make
        _ibl_keys_str = st.text_input(
            'Logistic input regressors (comma-separated)',
            value='c',
            key='ibl_logistic_inputs',
            help='Each name must be present under `data["inputs"]`. IBL data '
                 'includes `c` (signed contrast); add e.g. `reward, p_left` to '
                 'fit additional regressors with their own learnt weights.',
        )
        _ibl_keys = [k.strip() for k in _ibl_keys_str.split(',') if k.strip()] or ['c']
        (_ibl_llt, _, _ibl_K, _ibl_pnames, _ibl_dhyper, _, _, _ibl_data_spec) = (
            _ibl_logistic_make(_ibl_keys)
        )
        _ibl_race_fixed = False

    # --- Learning rule ---
    st.markdown('**Learning rule** *(optional)*')
    _ibl_lr_choice = st.selectbox(
        'Learning rule', ['None', 'REINFORCE (built-in)'], key='ibl_lr_choice',
    )

    _ibl_learning_rule = None
    _ibl_lr_reward_col = None

    if _ibl_lr_choice == 'REINFORCE (built-in)':
        from psytrax.learning_rules import augment_data_spec, make_reinforce
        _ibl_lr_reward_col = 'reward'  # always mapped from feedbackType
        _ibl_data_spec = augment_data_spec(_ibl_data_spec, make_reinforce(
            _ibl_llt, reward_key=_ibl_lr_reward_col))
        _ibl_learning_rule = make_reinforce(_ibl_llt, reward_key=_ibl_lr_reward_col)
        st.success('REINFORCE enabled — reward signal from IBL `feedbackType`.')

    # Validate data against model spec
    _ibl_needs_rt = 'rt' in _ibl_data_spec
    _ibl_spec_inputs = _ibl_data_spec.get('inputs', {})
    _ibl_missing = [k for k, info in _ibl_spec_inputs.items()
                    if info.get('required') and k not in raw['inputs']]
    if _ibl_missing:
        st.error(f'Data is missing required inputs: {_ibl_missing}')
        st.stop()
    if _ibl_needs_rt and 'times' not in raw:
        st.error('This model requires reaction times, but none are available.')
        st.stop()

    st.divider()

    # ==================================================================
    # 6. Configure fitting
    # ==================================================================
    st.subheader('6. Configure fitting')

    _fc1, _fc2 = st.columns(2)
    with _fc1:
        _ibl_ntrials = st.number_input(
            'Max trials (0 = all)', min_value=0, value=0, step=100, key='ibl_ntrials')
        _ibl_ntrials = int(_ibl_ntrials) if _ibl_ntrials > 0 else None

        _ibl_sess_bound = st.checkbox(
            'Session boundaries (fit sigDay)', value=True, key='ibl_session_boundaries')
        _ibl_shared_sigma = st.checkbox(
            'Shared sigma (scalar)', value=False, key='ibl_shared_sigma')

    with _fc2:
        _ibl_map_tol = st.select_slider(
            'MAP tolerance', options=[1e-3, 1e-4, 1e-5, 1e-6], value=1e-4,
            format_func=lambda x: f'{x:.0e}', key='ibl_map_tol')
        _ibl_hess = st.selectbox(
            'Credible intervals',
            ['All', 'weights', 'hyper', 'None'],
            index=0, key='ibl_hess',
            help='`All` computes credible intervals for both the trial-varying '
                 'parameters and the optimised hyperparameters; the hyper part '
                 'requires a small numerical Hessian.',
        )
        _ibl_hess = None if _ibl_hess == 'None' else _ibl_hess
        _ibl_precision = 'float64'

    with st.expander('Advanced: initial hyperparameters'):
        st.markdown('Leave blank to use model defaults. Values are in **log₂** scale.')
        _ibl_default_h = _ibl_dhyper()
        _ibl_sigma_str = st.text_input(
            f'sigma (log₂), {_ibl_K} values or single scalar',
            value=', '.join(f'{np.log2(v):.1f}' for v in np.atleast_1d(_ibl_default_h['sigma'])),
            key=f'ibl_sigma_init_{_ibl_model}',
        )
        try:
            _sv = [float(x.strip()) for x in _ibl_sigma_str.split(',')]
            _ibl_custom_sigma = float(2 ** _sv[0]) if len(_sv) == 1 else 2 ** np.array(_sv)
        except Exception:
            st.warning('Could not parse sigma — using model default.')
            _ibl_custom_sigma = _ibl_default_h['sigma']

    _ibl_hyper = _ibl_dhyper()
    if _ibl_shared_sigma:
        _ibl_hyper['sigma'] = (float(_ibl_custom_sigma) if np.isscalar(_ibl_custom_sigma)
                               else float(np.mean(_ibl_custom_sigma)))
    else:
        _ibl_hyper['sigma'] = (np.full(_ibl_K, float(_ibl_custom_sigma))
                               if np.isscalar(_ibl_custom_sigma)
                               else np.asarray(_ibl_custom_sigma))

    st.divider()

    # ==================================================================
    # 7. Fit
    # ==================================================================
    st.subheader('7. Fit')

    _ibl_run = st.button('Run fit', disabled=st.session_state['ibl_fit_running'], key='ibl_fit_run')

    if _ibl_run:
        import psytrax
        import psytrax._hyper_opt as _ibl_hyper_opt_mod
        import time as _time

        st.session_state['ibl_fit_running'] = True
        st.session_state['ibl_fit_result_path'] = None
        st.session_state['ibl_fit_log'] = []
        st.session_state['ibl_fit_error'] = None

        _iq = queue.Queue()

        class _IBLQueueTqdm:
            def __init__(self, *args, **kwargs):
                self._n = 0; self._map_n = 0; self._postfix = {}
            def update(self, n=1):
                self._n += n; self._map_n = 0
                self._postfix.pop('MAP loss', None)
                _iq.put(('progress', self._n, self._map_n, dict(self._postfix)))
            def set_postfix(self, d, **kwargs):
                self._postfix.update(d)
                if 'MAP loss' in d: self._map_n += 1
                _iq.put(('progress', self._n, self._map_n, dict(self._postfix)))
            def close(self): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass

        def _ibl_status_cb(payload):
            _iq.put(('status', payload))

        _ibl_orig_tqdm = _ibl_hyper_opt_mod.tqdm
        _ibl_hyper_opt_mod.tqdm = _IBLQueueTqdm

        def _ibl_run_fit():
            try:
                os.makedirs('fits', exist_ok=True)
                _subj = st.session_state.get('ibl_subject_selected') or \
                    st.session_state.get('ibl_subject_query', 'ibl_subject')
                fit_kw = dict(
                    data=raw,
                    shared_sigma=_ibl_shared_sigma,
                    session_boundaries=_ibl_sess_bound,
                    n_trials=_ibl_ntrials,
                    hess_calc=_ibl_hess,
                    map_tol=float(_ibl_map_tol),
                    precision=_ibl_precision,
                    subject_name=_subj,
                    save=True, verbose=True,
                    status_callback=_ibl_status_cb,
                )

                fit_kw.update(log_lik_trial=_ibl_llt, n_params=_ibl_K,
                              param_names=_ibl_pnames)

                if _ibl_learning_rule is not None:
                    fit_kw['learning_rule'] = _ibl_learning_rule
                    _ibl_status_cb({'stage': 'setup',
                                    'message': 'Learning rule enabled — alpha will be optimised.'})

                result = psytrax.fit(hyper=_ibl_hyper, **fit_kw)
                _iq.put(('done', result))
            except Exception:
                import traceback
                _iq.put(('error', traceback.format_exc()))
            finally:
                _ibl_hyper_opt_mod.tqdm = _ibl_orig_tqdm

        _ibl_thread = threading.Thread(target=_ibl_run_fit, daemon=True)
        _ibl_thread.start()
        st.session_state['_ibl_fit_thread'] = _ibl_thread
        st.session_state['_ibl_fit_queue'] = _iq

    if st.session_state['ibl_fit_running']:
        import time as _time
        _iq      = st.session_state['_ibl_fit_queue']
        _ithread = st.session_state['_ibl_fit_thread']

        st.markdown('**Fitting in progress…** &nbsp; `JAX L-BFGS`')
        _ic1, _ic2 = st.columns(2)
        _cyc_txt  = _ic1.empty()
        _map_txt  = _ic2.empty()
        _stat_txt = st.empty()
        _evd_txt  = st.empty()
        _log_box  = st.empty()

        _cyc, _map_i = 0, 0
        _evd_s, _best_s, _ml_s = '—', '—', '—'
        _cur_status = 'Preparing fit…'
        _flog = st.session_state.get('ibl_fit_log', [])

        while _ithread.is_alive():
            while not _iq.empty():
                try:
                    _m = _iq.get_nowait()
                    if _m[0] == 'progress':
                        _, _cyc, _map_i, _pf = _m
                        _evd_s = _pf.get('log_evd', '—')
                        _best_s = _pf.get('best', '—')
                        _ml_s = _pf.get('MAP loss', _ml_s)
                    elif _m[0] == 'status':
                        _cur_status = _m[1].get('message', _cur_status)
                        _flog.append(_cur_status)
                        _flog = _flog[-12:]
                        st.session_state['ibl_fit_log'] = _flog
                except queue.Empty:
                    break
            _cyc_txt.metric('Cycles completed', _cyc)
            _map_txt.metric('MAP iters (current cycle)', _map_i)
            _stat_txt.markdown(f'**Current step:** {_cur_status}')
            _evd_txt.markdown(
                f'Log evidence — current: **{_evd_s}** | best: **{_best_s}**'
                + (f' | Neg. log posterior: **{_ml_s}**' if _ml_s != '—' else ''))
            if _flog:
                _log_box.code('\n'.join(_flog), language='text')
            _time.sleep(0.5)

        # Drain queue
        _mtype, _mpay = 'error', 'No result received.'
        while not _iq.empty():
            try:
                _mm = _iq.get_nowait()
                if _mm[0] in ('done', 'error'):
                    _mtype, _mpay = _mm[0], _mm[1]
            except queue.Empty:
                break

        st.session_state['ibl_fit_running'] = False
        if _mtype == 'done':
            st.session_state['ibl_fit_result_path'] = _mpay
        else:
            st.session_state['ibl_fit_error'] = _mpay
        st.rerun()

    if st.session_state['ibl_fit_error']:
        st.error(f'Fitting failed:\n\n```\n{st.session_state["ibl_fit_error"]}\n```')

    if st.session_state['ibl_fit_result_path']:
        _ibl_path = st.session_state['ibl_fit_result_path']
        st.success(f'Fit complete! Saved to `{_ibl_path}`')

        _ibl_res = np.load(_ibl_path, allow_pickle=True).item()
        _r1, _r2, _r3, _r4 = st.columns(4)
        _r1.metric('Trials', _ibl_res['params'].shape[1])
        _r2.metric('Parameters', _ibl_res['params'].shape[0])
        _r3.metric('Log evidence', f"{_ibl_res['log_evidence']:.1f}")
        _r4.metric('Duration', str(_ibl_res['duration']).split('.')[0])

        with open(_ibl_path, 'rb') as _f:
            st.download_button(
                'Download fit file (.npy)', data=_f.read(),
                file_name=os.path.basename(_ibl_path),
                mime='application/octet-stream', key='ibl_download_fit',
            )

        st.divider()

        # ==============================================================
        # 8. Visualise results
        # ==============================================================
        st.subheader('8. Results')

        _ip = _ibl_res['params']
        _ipn = _ibl_res['param_names']
        _iK, _iN = _ip.shape
        _iWstd = _ibl_res['hess_info'].get('W_std')
        _idat = _ibl_res['data']

        _i_model_family, _i_fixed_params = _model_family_info(_ipn, result=_ibl_res)
        _i_rt_key = next((k for k in ('T', 'times') if k in _idat and _idat[k] is not None), None)
        _i_has_rt = (_i_rt_key is not None) and _i_model_family in _RT_CURVE_FAMILIES

        COLORS = ['#4e9af1', '#f1a44e', '#4ef17a', '#f14e7a', '#c44ef1', '#f1f14e']

        # --- Parameter trajectories ---
        st.markdown('#### Parameter trajectories')
        _i_traj_mode = st.radio('Display mode', ['Separate', 'Combined'],
                                horizontal=True, label_visibility='collapsed',
                                key='ibl_traj_mode')

        _i_trials = np.arange(_iN)
        _i_dl = _idat.get('dayLength') if _idat.get('dayLength') is not None else np.array([])
        _i_bounds = np.cumsum(_i_dl).astype(int) if len(_i_dl) else np.array([], dtype=int)

        if _i_traj_mode == 'Separate':
            _ncols = min(_iK, 3)
            _nrows = int(np.ceil(_iK / _ncols))
            fig, axes = plt.subplots(_nrows, _ncols,
                                     figsize=(5 * _ncols, 3 * _nrows), squeeze=False)
            _tc = _style_fig(fig)
            for k, (ax, name) in enumerate(zip(axes.flat, _ipn)):
                col = COLORS[k % len(COLORS)]
                _style_ax(ax, xlabel='Trial', title=name)
                ax.plot(_i_trials, _ip[k], color=col, lw=0.8, alpha=0.9)
                if _iWstd is not None:
                    ax.fill_between(_i_trials, _ip[k] - _iWstd[k], _ip[k] + _iWstd[k],
                                    color=col, alpha=0.2)
                for b in _i_bounds[:-1]:
                    ax.axvline(b, color=_tc['text'], lw=0.5, alpha=0.3, ls='--')
            for ax in axes.flat[_iK:]:
                ax.set_visible(False)
            fig.tight_layout()
            _show_fig(fig, 'ibl_param_trajectories.png')
        else:
            fig, ax = plt.subplots(figsize=(12, 4))
            _tc = _style_fig(fig)
            _style_ax(ax, xlabel='Trial', ylabel='Parameter value')
            for k, name in enumerate(_ipn):
                col = COLORS[k % len(COLORS)]
                ax.plot(_i_trials, _ip[k], color=col, lw=0.9, alpha=0.9, label=name)
                if _iWstd is not None:
                    ax.fill_between(_i_trials, _ip[k] - _iWstd[k], _ip[k] + _iWstd[k],
                                    color=col, alpha=0.15)
            for b in _i_bounds[:-1]:
                ax.axvline(b, color=_tc['text'], lw=0.5, alpha=0.3, ls='--')
            _style_legend(ax)
            fig.tight_layout()
            _show_fig(fig, 'ibl_param_trajectories.png')

        # --- Psychometric & chronometric curves ---
        if 'inputs' in _idat and 'c' in _idat['inputs'] and 'r' in _idat:
            c_data = _idat['inputs']['c']
            r_data = _idat['r']
            c_grid = np.linspace(c_data.min() - 0.05, c_data.max() + 0.05, 200)

            st.markdown('#### Psychometric curve: evolution over learning')
            N_WIN = 4
            edges = np.linspace(0, _iN, N_WIN + 1, dtype=int)

            fig_evo, axes_evo = plt.subplots(2, 2, figsize=(11, 8))
            _tc = _style_fig(fig_evo)
            for wi, ax in enumerate(axes_evo.flat):
                t0, t1 = int(edges[wi]), int(edges[wi + 1])
                mask = np.zeros(_iN, dtype=bool)
                mask[t0:t1] = True
                c_win, r_win = c_data[mask], r_data[mask]
                c_uniq = np.unique(c_win)
                p_win = np.array([r_win[c_win == cv].mean() for cv in c_uniq])
                n_win = np.array([np.sum(c_win == cv) for cv in c_uniq])

                _style_ax(ax, xlabel='Signed contrast', ylabel='P(right)',
                          title=f'Trials {t0 + 1}–{t1}')
                ax.scatter(c_uniq, p_win, s=[max(10, n / 5) for n in n_win],
                           color=_tc['text'], zorder=3)
                p_m, _ = _curve_predictions(
                    _ip[:, t0:t1], _ipn, c_grid, _i_model_family,
                    fixed_params=_i_fixed_params)
                if p_m is not None:
                    ax.plot(c_grid, p_m, color='#4e9af1', lw=2)
                ax.axhline(0.5, color=_tc['text'], lw=0.5, ls='--', alpha=0.4)
                ax.axvline(0, color=_tc['text'], lw=0.5, ls='--', alpha=0.4)
                ax.set_ylim(0, 1)
            fig_evo.suptitle('Psychometric curve evolution', color=_tc['text'], fontsize=13)
            fig_evo.tight_layout()
            _show_fig(fig_evo, 'ibl_psychometric_evolution.png')

            # --- Chronometric curves ---
            if _i_has_rt:
                T_data = _idat[_i_rt_key]
                st.markdown('#### Chronometric curve: evolution over learning')
                with st.spinner('Computing chronometric curves…'):
                    fig_cevo, axes_cevo = plt.subplots(2, 2, figsize=(11, 8))
                    _tc = _style_fig(fig_cevo)
                    panel_data = []
                    y_series = []
                    for wi in range(len(axes_cevo.flat)):
                        t0, t1 = int(edges[wi]), int(edges[wi + 1])
                        mask = np.zeros(_iN, dtype=bool)
                        mask[t0:t1] = True
                        c_win, rt_win = c_data[mask], T_data[mask]
                        c_uniq = np.unique(c_win)
                        rt_mean = np.array([rt_win[c_win == cv].mean() for cv in c_uniq])
                        n_win = np.array([np.sum(c_win == cv) for cv in c_uniq])
                        _, rt_m = _curve_predictions(
                            _ip[:, t0:t1], _ipn, c_grid, _i_model_family,
                            fixed_params=_i_fixed_params)
                        panel_data.append((t0, t1, c_uniq, rt_mean, n_win, rt_m))
                        y_series.extend([rt_mean, rt_m])

                    shared_ylim = _shared_ylim(y_series)
                    for ax, (t0, t1, c_uniq, rt_mean, n_win, rt_m) in zip(
                            axes_cevo.flat, panel_data):
                        _style_ax(ax, xlabel='Signed contrast', ylabel='Mean RT (s)',
                                  title=f'Trials {t0 + 1}–{t1}')
                        ax.scatter(c_uniq, rt_mean, s=[max(10, n / 5) for n in n_win],
                                   color=_tc['text'], zorder=3)
                        if rt_m is not None:
                            ax.plot(c_grid, rt_m, color='#4e9af1', lw=2)
                        ax.axvline(0, color=_tc['text'], lw=0.5, ls='--', alpha=0.4)
                        if shared_ylim is not None:
                            ax.set_ylim(*shared_ylim)
                    fig_cevo.suptitle('Chronometric curve evolution',
                                      color=_tc['text'], fontsize=13)
                    fig_cevo.tight_layout()
                    _show_fig(fig_cevo, 'ibl_chronometric_evolution.png')

        # --- Hyperparameter table ---
        st.markdown('#### Optimised hyperparameters (log₂ scale)')
        _i_hyper = _ibl_res['hyper']
        _hyp_rows = {}
        for _hk in ('sigma', 'sigInit', 'sigDay', 'alpha'):
            _hv = _i_hyper.get(_hk)
            if _hv is not None:
                _hv = np.atleast_1d(_hv)
                _hyp_rows[_hk] = {name: f'{np.log2(v):.2f}' for name, v in zip(_ipn, _hv)}
        if _hyp_rows:
            st.dataframe(pd.DataFrame(_hyp_rows).T, use_container_width=True)

# ---------------------------------------------------------------------------
# Visualise Results
# ---------------------------------------------------------------------------
elif page == 'Visualise Results':
    import os as _os
    st.title('Visualise Results')

    _fits_dir = _os.path.join(_os.path.dirname(__file__), 'example_fits')
    _example_fits = sorted(
        f.replace('_race_fit.npy', '')
        for f in _os.listdir(_fits_dir)
        if f.endswith('_race_fit.npy')
    ) if _os.path.isdir(_fits_dir) else []

    _vis_source = st.radio(
        'Data source',
        (['Example fits', 'Upload my own file'] if _example_fits else ['Upload my own file']),
        horizontal=True,
        key='vis_source',
    )

    if _vis_source == 'Example fits':
        _animal = st.selectbox('Select animal', _example_fits, key='vis_animal')
        result = np.load(
            _os.path.join(_fits_dir, f'{_animal}_race_fit.npy'), allow_pickle=True
        ).item()
    else:
        uploaded = st.file_uploader('Upload a psytrax fit file (.npy)', type='npy')
        if uploaded is None:
            st.info('Upload a `.npy` file saved by `psytrax.fit(..., save=True)` to visualise results.')
            st.stop()
        result = np.load(uploaded, allow_pickle=True).item()

    params      = result['params']          # (K, N)
    param_names = result['param_names']
    K, N        = params.shape
    W_std       = result['hess_info'].get('W_std')  # (K, N) or None
    log_evd     = result['log_evidence']
    hyper       = result['hyper']
    dat         = result['data']

    # --- Summary metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric('Trials', N)
    col2.metric('Parameters', K)
    col3.metric('Log evidence', f'{log_evd:.1f}')

    st.divider()

    # --- Model detection ---
    model_family, fixed_params = _model_family_info(param_names, result=result)
    # Locate RT array (stored as 'T' or 'times')
    _rt_key = next((k for k in ('T', 'times') if k in dat and dat[k] is not None), None)
    has_rt  = (_rt_key is not None) and model_family in _RT_CURVE_FAMILIES

    COLORS = ['#4e9af1', '#f1a44e', '#4ef17a', '#f14e7a', '#c44ef1', '#f1f14e']

    # --- Parameter trajectories ---
    st.subheader('Parameter trajectories')
    _render_parameter_trajectories(result, key_suffix='_vis')

    # --- Psychometric & chronometric curves ---
    if 'inputs' in dat and 'c' in dat['inputs'] and 'r' in dat:
        c_data = dat['inputs']['c']
        r_data = dat['r']
        contrasts_unique = np.unique(c_data)
        c_grid = np.linspace(contrasts_unique.min(), contrasts_unique.max(), 100)

        # --- Psychometric evolution ---
        st.subheader('Psychometric curve: evolution over learning')
        N_WIN = 4
        edges = np.linspace(0, N, N_WIN + 1, dtype=int)

        fig_evo, axes_evo = plt.subplots(2, 2, figsize=(11, 8))
        _tc = _style_fig(fig_evo)

        for wi, ax in enumerate(axes_evo.flat):
            t0, t1 = int(edges[wi]), int(edges[wi + 1])
            mask   = np.zeros(N, dtype=bool)
            mask[t0:t1] = True

            c_win = c_data[mask]
            r_win = r_data[mask]
            c_uniq_win = np.unique(c_win)
            p_win  = np.array([r_win[c_win == cv].mean() for cv in c_uniq_win])
            n_win  = np.array([np.sum(c_win == cv) for cv in c_uniq_win])

            _style_ax(ax, xlabel='Signed contrast', ylabel='P(right)',
                      title=f'Trials {t0 + 1}–{t1}')
            ax.scatter(c_uniq_win, p_win, s=[max(10, n / 5) for n in n_win],
                       color=_tc['text'], zorder=3)

            params_win = params[:, t0:t1]
            p_m, _ = _curve_predictions(
                params_win, param_names, c_grid, model_family, fixed_params=fixed_params
            )
            if p_m is not None:
                ax.plot(c_grid, p_m, color='#4e9af1', lw=2)

            ax.axhline(0.5, color=_tc['text'], lw=0.5, ls='--', alpha=0.4)
            ax.axvline(0,   color=_tc['text'], lw=0.5, ls='--', alpha=0.4)
            ax.set_ylim(0, 1)

        fig_evo.suptitle('Psychometric curve evolution', color=_tc['text'], fontsize=13)
        fig_evo.tight_layout()
        _show_fig(fig_evo, 'psychometric_evolution.png')

        # --- Chronometric evolution (RT-capable models only) ---
        if has_rt:
            T_data = dat[_rt_key]
            st.subheader('Chronometric curve: evolution over learning')
            with st.spinner('Computing chronometric curves…'):
                fig_cevo, axes_cevo = plt.subplots(2, 2, figsize=(11, 8))
                _tc = _style_fig(fig_cevo)
                panel_data = []
                y_series = []

                for wi in range(len(axes_cevo.flat)):
                    t0, t1 = int(edges[wi]), int(edges[wi + 1])
                    mask   = np.zeros(N, dtype=bool)
                    mask[t0:t1] = True

                    c_win  = c_data[mask]
                    rt_win = T_data[mask]
                    c_uniq_win  = np.unique(c_win)
                    rt_win_mean = np.array([rt_win[c_win == cv].mean() for cv in c_uniq_win])
                    n_win       = np.array([np.sum(c_win == cv) for cv in c_uniq_win])
                    _, rt_m = _curve_predictions(
                        params[:, t0:t1], param_names, c_grid, model_family,
                        fixed_params=fixed_params
                    )
                    panel_data.append((t0, t1, c_uniq_win, rt_win_mean, n_win, rt_m))
                    y_series.extend([rt_win_mean, rt_m])

                shared_ylim = _shared_ylim(y_series)

                for ax, (t0, t1, c_uniq_win, rt_win_mean, n_win, rt_m) in zip(axes_cevo.flat, panel_data):
                    _style_ax(ax, xlabel='Signed contrast', ylabel='Mean RT (s)',
                              title=f'Trials {t0 + 1}–{t1}')
                    ax.scatter(c_uniq_win, rt_win_mean, s=[max(10, n / 5) for n in n_win],
                               color=_tc['text'], zorder=3)
                    ax.plot(c_grid, rt_m, color='#4e9af1', lw=2)
                    ax.axvline(0, color=_tc['text'], lw=0.5, ls='--', alpha=0.4)
                    if shared_ylim is not None:
                        ax.set_ylim(*shared_ylim)

                fig_cevo.suptitle('Chronometric curve evolution', color=_tc['text'], fontsize=13)
                fig_cevo.tight_layout()
                _show_fig(fig_cevo, 'chronometric_evolution.png')

        # --- Dopamine evolution (race only, when data has dopamine) ----
        _da_data = dat.get('dopamine') if isinstance(dat, dict) else None
        _model_mh = result.get('model_hyper') or {}
        _has_dopamine = (
            model_family == 'race'
            and _da_data is not None
            and np.asarray(_da_data).shape[0] == N
        )
        if _has_dopamine:
            st.subheader('Dopamine: empirical vs predicted, by quartile')
            st.caption(
                'Black dots: empirical mean dopamine peak per signed contrast '
                '(within each quartile of trials). Blue line: analytic '
                'prediction `tanh(0.5 · da_beta · (w_eff · |c| − da_offset))` '
                'averaged over each window\'s '
                'recovered trajectory, where `w_eff = wr` for `c ≥ 0` and '
                '`w_eff = wl` otherwise.'
            )
            da_data = np.asarray(_da_data, dtype=float)
            try:
                _wr_idx = list(param_names).index('wr')
                _wl_idx = list(param_names).index('wl')
            except ValueError:
                _wr_idx = _wl_idx = None

            if _wr_idx is not None:
                fig_da, axes_da = plt.subplots(2, 2, figsize=(11, 8))
                _tc = _style_fig(fig_da)
                panel_da = []
                y_da = []
                for wi in range(N_WIN):
                    t0, t1 = int(edges[wi]), int(edges[wi + 1])
                    c_win   = c_data[t0:t1]
                    da_win  = da_data[t0:t1]
                    finite  = np.isfinite(da_win)
                    c_win_f = c_win[finite]
                    da_win_f = da_win[finite]
                    c_uniq_win = np.unique(c_win_f) if c_win_f.size else np.array([])
                    da_mean = np.array([
                        da_win_f[c_win_f == cv].mean() for cv in c_uniq_win
                    ]) if c_uniq_win.size else np.array([])
                    da_sem = np.array([
                        (da_win_f[c_win_f == cv].std() /
                         max(np.sqrt(np.sum(c_win_f == cv)), 1.0))
                        for cv in c_uniq_win
                    ]) if c_uniq_win.size else np.array([])
                    n_w = np.array([np.sum(c_win_f == cv) for cv in c_uniq_win])

                    # Analytic curve: average the tanh dopamine readout
                    # over the window. β and offset come from the recovered
                    # model_hyper (EB-fitted alongside sig_DA) when present,
                    # otherwise fall back to the model defaults imported at
                    # module top.
                    _DA_BETA   = float(_model_mh.get('da_beta',   _DA_BETA_DEFAULT))
                    _DA_OFFSET = float(_model_mh.get('da_offset', _DA_OFFSET_DEFAULT))
                    wr_win = params[_wr_idx, t0:t1]
                    wl_win = params[_wl_idx, t0:t1]
                    da_curve = np.array([
                        np.mean(_dopamine_tanh_readout(
                            np.where(cv >= 0, wr_win, wl_win) * abs(cv),
                            _DA_BETA,
                            _DA_OFFSET,
                        ))
                        for cv in c_grid
                    ])
                    panel_da.append((t0, t1, c_uniq_win, da_mean, da_sem,
                                     n_w, da_curve))
                    y_da.extend([da_mean, da_curve])

                shared_ylim_da = _shared_ylim(y_da)
                for ax, (t0, t1, c_uniq_win, da_mean, da_sem,
                         n_w, da_curve) in zip(axes_da.flat, panel_da):
                    _style_ax(ax, xlabel='Signed contrast',
                              ylabel='Dopamine peak (a.u.)',
                              title=f'Trials {t0 + 1}–{t1}')
                    if c_uniq_win.size:
                        ax.errorbar(
                            c_uniq_win, da_mean, yerr=da_sem,
                            fmt='o', color=_tc['text'], ecolor=_tc['text'],
                            elinewidth=0.8, capsize=2, markersize=5,
                            zorder=3,
                        )
                    ax.plot(c_grid, da_curve, color='#4e9af1', lw=2)
                    ax.axvline(0, color=_tc['text'], lw=0.5, ls='--', alpha=0.4)
                    if shared_ylim_da is not None:
                        ax.set_ylim(*shared_ylim_da)

                if _model_mh.get('sig_DA') is not None:
                    _title_bits = [
                        f'σ_DA = {_model_mh["sig_DA"]:.3f}',
                        f'β = {_DA_BETA:.2f}',
                        f'centre = {_DA_OFFSET:.3f}',
                    ]
                    fig_da.suptitle(
                        'Dopamine: empirical vs predicted   '
                        '(' + ',  '.join(_title_bits) + ')',
                        color=_tc['text'], fontsize=13,
                    )
                else:
                    fig_da.suptitle(
                        'Dopamine: empirical vs predicted',
                        color=_tc['text'], fontsize=13,
                    )
                fig_da.tight_layout()
                _show_fig(fig_da, 'dopamine_evolution.png')
            else:
                st.info(
                    'Dopamine field detected but the fit\'s parameter names '
                    'don\'t include `wr`, `wl`, `z` — skipping the dopamine '
                    'plot.'
                )

    # --- Hyperparameter table ---
    st.subheader('Optimised hyperparameters (log₂ scale)')
    import pandas as pd
    hyper_rows = []
    for key, val in hyper.items():
        if val is not None:
            arr = np.atleast_1d(val)
            row = {'hyperparameter': key}
            if len(arr) == 1:
                row['shared'] = f'{np.log2(arr[0]):.3f}'
            else:
                row.update({
                    name: f'{np.log2(arr[i]):.3f}'
                    for i, name in enumerate(param_names)
                    if i < len(arr)
                })
            hyper_rows.append(row)
    if hyper_rows:
        st.dataframe(pd.DataFrame(hyper_rows).set_index('hyperparameter'), use_container_width=True)

    # --- Hyperparameter credible intervals (Laplace, log₂ scale) ---
    _hyp_std = (result.get('hess_info') or {}).get('hyp_std')
    if _hyp_std is not None:
        # The hyp_std vector is laid out the same way as _pack_optvals: hyper
        # entries first (each contributing 1 if scalar or K if vector), then
        # one entry per optimised model_hyper key.
        _hyp_optList = (result.get('hess_info') or {}).get('hyp_optList') or []
        _hyp_mh_optList = (result.get('hess_info') or {}).get('hyp_model_hyper_optList') or []
        K = len(param_names)
        _ci_rows = []
        _idx = 0
        _hyp_arr = np.asarray(_hyp_std)
        for key in _hyp_optList:
            val = hyper.get(key)
            if val is None:
                continue
            arr = np.atleast_1d(val)
            row = {'hyperparameter': key}
            if len(arr) == 1:
                std = float(_hyp_arr[_idx]) if _idx < len(_hyp_arr) else float('nan')
                row['shared (±1σ)'] = '—' if not np.isfinite(std) else f'±{std:.3f}'
                _idx += 1
            else:
                for i, name in enumerate(param_names):
                    if i >= len(arr):
                        break
                    std = float(_hyp_arr[_idx]) if _idx < len(_hyp_arr) else float('nan')
                    row[name] = '—' if not np.isfinite(std) else f'±{std:.3f}'
                    _idx += 1
            _ci_rows.append(row)
        for key in _hyp_mh_optList:
            std = float(_hyp_arr[_idx]) if _idx < len(_hyp_arr) else float('nan')
            _ci_rows.append({
                'hyperparameter': f'model_hyper[{key}]',
                'shared (±1σ)': '—' if not np.isfinite(std) else f'±{std:.3f}',
            })
            _idx += 1
        if _ci_rows:
            st.caption(
                'Credible intervals for the optimised hyperparameters '
                '(±1 SD, log₂ scale, from the Laplace approximation). '
                'A value of `—` means the numerical Hessian was degenerate '
                'in that direction (typically because the hyperparameter '
                'sat at a bound or was confounded with another).'
            )
            st.dataframe(pd.DataFrame(_ci_rows).set_index('hyperparameter'),
                         use_container_width=True)
    elif (result.get('hess_info') or {}).get('hyp_std_error'):
        st.caption(
            f'Hyperparameter credible intervals could not be computed: '
            f'`{result["hess_info"]["hyp_std_error"]}`. The fit itself is fine — '
            'this just means the numerical Hessian was singular at the optimum.'
        )

    # --- Model-level hyperparameters (linear scale, one scalar per entry) ---
    _model_hyper = result.get('model_hyper') or {}
    if _model_hyper:
        st.subheader('Optimised model-level hyperparameters')
        st.caption('These are constants the model exposes to Empirical Bayes (e.g. the race model\'s within-trial accumulator noise `sig_i`).')
        st.dataframe(
            pd.DataFrame(
                [{'hyperparameter': k, 'value': float(v)} for k, v in _model_hyper.items()]
            ).set_index('hyperparameter'),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Compare Models
# ---------------------------------------------------------------------------
elif page == 'Compare Models':
    import os as _os
    import pandas as pd
    st.title('Compare Models')

    _fits_dir_cmp = _os.path.join(_os.path.dirname(__file__), 'example_fits')
    _example_fits_cmp = sorted(
        f.replace('_race_fit.npy', '')
        for f in _os.listdir(_fits_dir_cmp)
        if f.endswith('_race_fit.npy')
    ) if _os.path.isdir(_fits_dir_cmp) else []

    _cmp_source = st.radio(
        'Data source',
        (['Example fits', 'Upload my own files'] if _example_fits_cmp else ['Upload my own files']),
        horizontal=True,
        key='cmp_source',
    )

    if _cmp_source == 'Example fits':
        _selected_animals = st.multiselect(
            'Select animals to compare',
            _example_fits_cmp,
            default=_example_fits_cmp[:min(4, len(_example_fits_cmp))],
            key='cmp_animals',
        )
        if len(_selected_animals) < 2:
            st.info('Select at least two animals to compare.')
            st.stop()
        results = {
            animal: np.load(
                _os.path.join(_fits_dir_cmp, f'{animal}_race_fit.npy'), allow_pickle=True
            ).item()
            for animal in _selected_animals
        }
    else:
        uploaded_files = st.file_uploader(
            'Upload multiple psytrax fit files (.npy)',
            type='npy',
            accept_multiple_files=True,
        )
        if not uploaded_files:
            st.info('Upload two or more `.npy` fit files (e.g. DAP009_logistic_fit.npy, DAP009_race_fit.npy …)')
            st.stop()
        results = {}
        for f in uploaded_files:
            res  = np.load(f, allow_pickle=True).item()
            name = f.name.replace('.npy', '')
            results[name] = res

    # --- Log evidence bar chart ---
    st.subheader('Log evidence (higher = better fit)')
    names  = list(results.keys())
    evds   = [results[n]['log_evidence'] for n in names]

    fig, ax = plt.subplots(figsize=(max(4, len(names) * 1.2), 4))
    _tc = _style_fig(fig)
    _style_ax(ax, ylabel='Log evidence')
    colors = ['#4e9af1', '#f1a44e', '#4ef17a', '#f14e7a', '#c44ef1']
    bars = ax.bar(names, evds, color=[colors[i % len(colors)] for i in range(len(names))])
    ax.bar_label(bars, fmt='%.1f', color=_tc['text'], padding=4)
    ax.set_xticklabels(names, rotation=20, ha='right')
    fig.tight_layout()
    _show_fig(fig, 'log_evidence.png')

    # --- Summary table ---
    rows = []
    for n, res in results.items():
        rows.append({
            'Model':         n,
            'K (params)':    res['params'].shape[0],
            'N (trials)':    res['params'].shape[1],
            'Log evidence':  f"{res['log_evidence']:.2f}",
            'Duration':      str(res['duration']).split('.')[0],
        })
    st.dataframe(pd.DataFrame(rows).set_index('Model'), use_container_width=True)

    # --- Psychometric & chronometric evolution per model ---
    first_res = next(iter(results.values()))
    dat = first_res['data']

    if 'inputs' in dat and 'c' in dat['inputs'] and 'r' in dat:
        c_data = dat['inputs']['c']
        r_data = dat['r']
        contrasts_unique = np.unique(c_data)
        c_grid = np.linspace(contrasts_unique.min(), contrasts_unique.max(), 100)
        N_cm   = first_res['params'].shape[1]
        N_WIN  = 4
        edges  = np.linspace(0, N_cm, N_WIN + 1, dtype=int)

        # Detect if any model has RT data
        _rt_key_cm = next((k for k in ('T', 'times') if k in dat and dat[k] is not None), None)
        model_families = {
            name: _model_family_info(res['param_names'], result=res)[0]
            for name, res in results.items()
        }
        any_rt_model = any(family in _RT_CURVE_FAMILIES for family in model_families.values())

        # --- Psychometric evolution ---
        st.subheader('Psychometric curve: evolution over learning')
        with st.spinner('Computing psychometric curves…'):
            fig_p, axes_p = plt.subplots(2, 2, figsize=(11, 8))
            _tc = _style_fig(fig_p)

            for wi, ax in enumerate(axes_p.flat):
                t0, t1 = int(edges[wi]), int(edges[wi + 1])
                mask   = np.zeros(N_cm, dtype=bool)
                mask[t0:t1] = True

                c_win = c_data[mask]; r_win = r_data[mask]
                c_uniq_win = np.unique(c_win)
                p_win = np.array([r_win[c_win == cv].mean() for cv in c_uniq_win])
                n_win = np.array([np.sum(c_win == cv) for cv in c_uniq_win])

                _style_ax(ax, xlabel='Signed contrast', ylabel='P(right)',
                          title=f'Trials {t0 + 1}–{t1}')
                ax.scatter(c_uniq_win, p_win, s=[max(10, n / 5) for n in n_win],
                           color=_tc['text'], zorder=3, label='data')

                for mi, (mname, res) in enumerate(results.items()):
                    pn  = res['param_names']
                    par = res['params'][:, t0:t1]
                    col = colors[mi % len(colors)]
                    family, fixed_params = _model_family_info(pn, result=res)
                    p_m, _ = _curve_predictions(
                        par, pn, c_grid, family, fixed_params=fixed_params
                    )
                    if p_m is not None:
                        ax.plot(c_grid, p_m, color=col, lw=2, label=mname)

                ax.axhline(0.5, color=_tc['text'], lw=0.5, ls='--', alpha=0.4)
                ax.axvline(0,   color=_tc['text'], lw=0.5, ls='--', alpha=0.4)
                ax.set_ylim(0, 1)
                _style_legend(ax, fontsize=7)

            fig_p.suptitle('Psychometric curve evolution', color=_tc['text'], fontsize=13)
            fig_p.tight_layout()
            _show_fig(fig_p, 'psychometric_comparison.png')

        # --- Chronometric evolution (RT-capable models + RT data) ---
        if _rt_key_cm is not None and any_rt_model:
            T_data = dat[_rt_key_cm]
            st.subheader('Chronometric curve: evolution over learning')
            with st.spinner('Computing chronometric curves…'):
                fig_c, axes_c = plt.subplots(2, 2, figsize=(11, 8))
                _tc = _style_fig(fig_c)
                panel_data = []
                y_series = []

                for wi in range(len(axes_c.flat)):
                    t0, t1 = int(edges[wi]), int(edges[wi + 1])
                    mask   = np.zeros(N_cm, dtype=bool)
                    mask[t0:t1] = True

                    c_win  = c_data[mask]; rt_win = T_data[mask]
                    c_uniq_win  = np.unique(c_win)
                    rt_win_mean = np.array([rt_win[c_win == cv].mean() for cv in c_uniq_win])
                    n_win       = np.array([np.sum(c_win == cv) for cv in c_uniq_win])
                    model_curves = []
                    for mi, (mname, res) in enumerate(results.items()):
                        pn  = res['param_names']
                        par = res['params'][:, t0:t1]
                        family, fixed_params = _model_family_info(pn, result=res)
                        if family in _RT_CURVE_FAMILIES:
                            _, rt_m = _curve_predictions(
                                par, pn, c_grid, family, fixed_params=fixed_params
                            )
                            model_curves.append((mname, colors[mi % len(colors)], rt_m))
                            y_series.append(rt_m)
                    panel_data.append((t0, t1, c_uniq_win, rt_win_mean, n_win, model_curves))
                    y_series.append(rt_win_mean)

                shared_ylim = _shared_ylim(y_series)

                for ax, (t0, t1, c_uniq_win, rt_win_mean, n_win, model_curves) in zip(axes_c.flat, panel_data):
                    _style_ax(ax, xlabel='Signed contrast', ylabel='Mean RT (s)',
                              title=f'Trials {t0 + 1}–{t1}')
                    ax.scatter(c_uniq_win, rt_win_mean, s=[max(10, n / 5) for n in n_win],
                               color=_tc['text'], zorder=3, label='data')
                    for mname, color, rt_m in model_curves:
                        ax.plot(c_grid, rt_m, color=color, lw=2, label=mname)
                    ax.axvline(0, color=_tc['text'], lw=0.5, ls='--', alpha=0.4)
                    if shared_ylim is not None:
                        ax.set_ylim(*shared_ylim)
                    _style_legend(ax, fontsize=7)

                fig_c.suptitle('Chronometric curve evolution', color=_tc['text'], fontsize=13)
                fig_c.tight_layout()
                _show_fig(fig_c, 'chronometric_comparison.png')

    # --- Parameter trajectories per model ---
    st.subheader('Parameter trajectories')
    traj_mode_cm = st.radio('Display mode', ['Separate', 'Combined'],
                            horizontal=True, label_visibility='collapsed',
                            key='traj_mode_cm')
    for i, (name, res) in enumerate(results.items()):
        with st.expander(name, expanded=(i == 0)):
            params      = res['params']
            param_names = res['param_names']
            K, N        = params.shape
            W_std       = res['hess_info'].get('W_std')
            trials      = np.arange(N)
            color       = colors[i % len(colors)]

            _dl = res['data'].get('dayLength')
            day_lengths = _dl if _dl is not None else np.array([])
            boundaries  = np.cumsum(day_lengths).astype(int) if len(day_lengths) else np.array([], dtype=int)

            if traj_mode_cm == 'Separate':
                n_cols = min(K, 3)
                n_rows = int(np.ceil(K / n_cols))
                fig3, axes = plt.subplots(n_rows, n_cols,
                                          figsize=(5 * n_cols, 3 * n_rows),
                                          squeeze=False)
                _tc = _style_fig(fig3)
                for k, (ax, pname) in enumerate(zip(axes.flat, param_names)):
                    _style_ax(ax, xlabel='Trial', title=pname)
                    ax.plot(trials, params[k], color=color, lw=0.8, alpha=0.9)
                    if W_std is not None:
                        ax.fill_between(trials, params[k] - W_std[k], params[k] + W_std[k],
                                        color=color, alpha=0.2)
                    for b in boundaries[:-1]:
                        ax.axvline(b, color=_tc['text'], lw=0.5, alpha=0.3, ls='--')
                for ax in axes.flat[K:]:
                    ax.set_visible(False)
                fig3.tight_layout(pad=1.3)
                _show_fig(fig3, f'params_{name}.png')
            else:  # Combined
                fig3, ax3 = plt.subplots(figsize=(12, 4))
                _tc = _style_fig(fig3)
                _style_ax(ax3, xlabel='Trial', ylabel='Parameter value')
                for k, pname in enumerate(param_names):
                    col_k = colors[k % len(colors)]
                    ax3.plot(trials, params[k], color=col_k, lw=0.9, alpha=0.9, label=pname)
                    if W_std is not None:
                        ax3.fill_between(trials, params[k] - W_std[k], params[k] + W_std[k],
                                         color=col_k, alpha=0.15)
                for b in boundaries[:-1]:
                    ax3.axvline(b, color=_tc['text'], lw=0.5, alpha=0.3, ls='--')
                _style_legend(ax3, fontsize=7)
                fig3.tight_layout(pad=1.3)
                _show_fig(fig3, f'params_{name}.png')


# ---------------------------------------------------------------------------
# Model Recovery
# ---------------------------------------------------------------------------
elif page == 'Model Recovery':
    import psytrax
    import psytrax._hyper_opt as _rec_hyper_opt_mod
    import importlib.util, tempfile
    import time as _rec_time
    import pandas as pd

    st.title('Model Recovery')
    st.markdown(
        'Generate parameter trajectories with sliders, simulate trial-by-trial '
        'data through any model, then run `psytrax.fit` and overlay the '
        'recovered trajectories on the truth.'
    )
    st.markdown(
        'This is the cleanest way to sanity-check a model: if the EB fit cannot '
        'recover trajectories you yourself injected, the same fit on real data '
        'is unlikely to be informative.'
    )
    st.divider()

    # ------------------------------------------------------------------
    # 1. Choose model
    # ------------------------------------------------------------------
    st.subheader('1. Choose model')
    _rec_model_choice = st.selectbox(
        'Model',
        ['Race model (inverse-Gaussian)',
         'DDM — exact (Navarro & Fuss 2009)',
         'Logistic regression',
         'Upload custom (.py)'],
        key='rec_model_choice',
    )

    def _load_builtin_model(name, **kwargs):
        if name == 'Race model (inverse-Gaussian)':
            from psytrax.models import race as _m
            _race_dh = (
                _m.default_model_hyper_with_dopamine
                if kwargs.get('race_with_dopamine')
                else _m.default_model_hyper
            )
            _da_blurb = (
                ' Joint dopamine fit is enabled: each trial also emits a '
                '`dopamine` value drawn from '
                '`N(tanh(0.5 · da_beta · (w_eff · |c| − da_offset)), sig_DA²)` '
                '(`w_eff = wr` if `c ≥ 0` else `wl`), and the fit estimates '
                '`sig_DA`, `da_beta`, and `da_offset` jointly with `sig_i`.'
                if kwargs.get('race_with_dopamine') else ''
            )
            return {
                'log_lik_trial': _m.log_lik_trial, 'sample_trial': _m.sample_trial,
                'N_PARAMS': _m.N_PARAMS, 'PARAM_NAMES': list(_m.PARAM_NAMES),
                'default_hyper': _m.default_hyper, 'default_E0': _m.default_E0,
                'default_model_hyper': _race_dh,
                'DATA_SPEC': _m.DATA_SPEC,
                'family': 'race',
                'desc': f"""
**Race model** — two independent inverse-Gaussian accumulators racing to a shared
threshold *z*. The first accumulator to hit threshold determines the choice; its
first-passage time is the reaction time. The five trial-varying parameters are
right/left drift weights (`wr`, `wl`), right/left baseline drifts (`br`, `bl`),
and the shared threshold `z`. The within-trial accumulator noise `sig_i` is a
model-level scalar estimated by Empirical Bayes alongside the random-walk noise
`sigma`. Inputs: signed contrast `c`. Outputs: choice + RT.{_da_blurb}
""",
            }
        if name == 'DDM — exact (Navarro & Fuss 2009)':
            from psytrax.models import ddm as _m
            return {
                'log_lik_trial': _m.log_lik_trial, 'sample_trial': _m.sample_trial,
                'N_PARAMS': _m.N_PARAMS, 'PARAM_NAMES': list(_m.PARAM_NAMES),
                'default_hyper': _m.default_hyper, 'default_E0': _m.default_E0,
                'default_model_hyper': lambda: {},
                'DATA_SPEC': _m.DATA_SPEC, 'family': 'ddm_exact',
                'desc': """
**DDM (exact)** — Wiener process between two absorbing barriers, with a fully
analytic likelihood (Navarro & Fuss 2009). Three parameters: `w` (contrast
weight), `b` (drift bias), `a` (boundary separation, > 0). Sampling integrates
a Wiener process with `dt = 1 ms`, so the simulator runs slower than the
inverse-Gaussian models. Inputs: `c`. Outputs: choice + RT.
""",
            }
        if name == 'Logistic regression':
            from psytrax.models.logistic import make_model as _logistic_make
            input_keys = list(kwargs.get('logistic_keys') or ['c'])
            (llt, samp, K, pnames, dh, dE0, _dlr, dspec) = _logistic_make(input_keys)
            weight_summary = ', '.join(f'`{k}`' for k in input_keys)
            return {
                'log_lik_trial': llt, 'sample_trial': samp,
                'N_PARAMS': K, 'PARAM_NAMES': list(pnames),
                'default_hyper': dh, 'default_E0': dE0,
                'default_model_hyper': lambda: {},
                'DATA_SPEC': dspec, 'family': 'logistic',
                'desc': f"""
**Logistic regression** — `P(right) = σ(w · x + b)`. {K} trial-varying parameters:
weights for {weight_summary}, plus a bias `b`. No reaction times — only choice
is modelled.
""",
            }
        return None

    # Logistic gets a comma-separated list of input regressors so the user
    # can fit additional features beyond the default contrast `c`.
    _logistic_keys_for_recovery = ['c']
    if _rec_model_choice == 'Logistic regression':
        _rec_logistic_keys_str = st.text_input(
            'Input regressors (comma-separated)',
            value='c',
            key='rec_logistic_inputs',
            help='Each name becomes a key under `data["inputs"][k]` and gets '
                 'its own trial-varying weight. The recovery page will ask '
                 'for a value pool for each one in the trial setup section.',
        )
        _logistic_keys_for_recovery = (
            [k.strip() for k in _rec_logistic_keys_str.split(',') if k.strip()]
            or ['c']
        )

    # Race model can additionally fit a per-trial dopamine peak, modelled as
    # N(tanh(0.5·da_beta·(w_eff·|c| − da_offset)), sig_DA²), with all three
    # dopamine readout scalars estimated by EB.
    _rec_race_with_dopamine = False
    if _rec_model_choice == 'Race model (inverse-Gaussian)':
        _rec_race_with_dopamine = st.checkbox(
            'Include dopamine signal (joint choice + RT + dopamine fit)',
            value=False,
            key='rec_race_with_dopamine',
            help='Add a per-trial dopamine peak to the simulated data and '
                 'fit it with a Gaussian likelihood whose mean is '
                 'tanh(0.5 · da_beta · (w_eff · |c| − da_offset)) '
                 '(w_eff = wr if c ≥ 0 else wl) and whose variance (sig_DA²) '
                 'is estimated jointly with sig_i.',
        )

    _rec_bundle = None
    if _rec_model_choice == 'Upload custom (.py)':
        st.markdown(
            'Upload a `.py` module that exposes:\n\n'
            '- `log_lik_trial(params, dat_trial, model_hyper)` — JAX-traceable\n'
            '- `sample_trial(params, dat_trial, rng, model_hyper)` — numpy sampler\n'
            '- `N_PARAMS`, `PARAM_NAMES` — number/names of the trial-varying parameters\n\n'
            'Optional: `default_hyper`, `default_E0`, `default_model_hyper`, `DATA_SPEC`. '
            'When a `default_E0(N)` is provided, the trajectory sliders below default '
            'to its first/last values; otherwise they default to zero.'
        )
        _up = st.file_uploader('Model module (.py)', type=['py'], key='rec_model_upload')
        if _up is not None:
            try:
                _src = _up.read().decode('utf-8')
                with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as _tmp:
                    _tmp.write(_src)
                    _tmp_path = _tmp.name
                _spec = importlib.util.spec_from_file_location('_user_recovery_model', _tmp_path)
                _mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                missing = [a for a in ('log_lik_trial', 'sample_trial', 'N_PARAMS')
                           if not hasattr(_mod, a)]
                if missing:
                    raise ValueError(
                        f'Module is missing required attribute(s): {", ".join(missing)}'
                    )
                _Knames = list(getattr(
                    _mod, 'PARAM_NAMES',
                    [str(i) for i in range(int(_mod.N_PARAMS))]
                ))
                _rec_bundle = {
                    'log_lik_trial': _mod.log_lik_trial,
                    'sample_trial':  _mod.sample_trial,
                    'N_PARAMS':      int(_mod.N_PARAMS),
                    'PARAM_NAMES':   _Knames,
                    'default_hyper': getattr(_mod, 'default_hyper', None),
                    'default_E0':    getattr(_mod, 'default_E0', None),
                    'default_model_hyper': getattr(_mod, 'default_model_hyper', lambda: {}),
                    'DATA_SPEC':     getattr(_mod, 'DATA_SPEC', None),
                    'family':        'custom',
                    'desc': (
                        f'**Custom model** loaded from `{_up.name}` — '
                        f'{int(_mod.N_PARAMS)} parameter(s): `{", ".join(_Knames)}`.'
                    ),
                }
                st.success(f'Loaded `{_up.name}` ({int(_mod.N_PARAMS)} params).')
            except Exception as exc:
                st.error(f'Failed to load model: {exc}')
        else:
            st.info('Upload a Python file to continue.')
    else:
        _rec_bundle = _load_builtin_model(
            _rec_model_choice,
            logistic_keys=_logistic_keys_for_recovery,
            race_with_dopamine=_rec_race_with_dopamine,
        )

    if _rec_bundle is None:
        st.stop()

    st.markdown(_rec_bundle['desc'])
    st.divider()

    # ------------------------------------------------------------------
    # 2. Trial setup — N, seed, value pools for the model's inputs
    # ------------------------------------------------------------------
    st.subheader('2. Trial setup')
    col_n, col_seed = st.columns([2, 1])
    with col_n:
        N_rec = st.slider('Number of trials', 200, 4000, 1000, step=100,
                          key='rec_n_trials',
                          help='More trials → better recovery, longer fit.')
    with col_seed:
        seed_rec = st.number_input('Random seed', min_value=0, max_value=2**31 - 1,
                                   value=42, step=1, key='rec_seed')

    _is_logistic = _rec_bundle['family'] == 'logistic'
    if _is_logistic:
        st.markdown(
            '**Trial inputs.** Every trial needs values for each input '
            'regressor. For each one, choose whether to draw values from a '
            'discrete pool or uniformly from a continuous range. A wide '
            'spread gives the EB fit more leverage for parameter identification.'
        )
    else:
        st.markdown(
            '**Trial inputs.** Every trial needs values for each of the model\'s '
            'inputs — for the race / DDM models this is the signed stimulus contrast '
            '`c` that the accumulators integrate. List the values you want to expose '
            'the model to below; one is sampled uniformly per trial. Choosing a '
            'wide range gives the EB fit more leverage for parameter identification.'
        )

    _spec = _rec_bundle.get('DATA_SPEC') or {}
    _rec_input_keys = list((_spec.get('inputs') or {}).keys()) or ['c']
    _DEFAULT_VALUE_POOLS = {
        'c':      '-1, -0.5, -0.25, 0, 0.25, 0.5, 1',
        'reward': '0, 1',
    }
    _DEFAULT_RANGES = {
        'c':      '-1, 1',
        'reward': '0, 1',
    }

    def _parse_discrete(_key, _vals_str):
        try:
            _v = np.asarray([float(x) for x in _vals_str.split(',') if x.strip()],
                            dtype=float)
            if _v.size == 0:
                raise ValueError
            return _v
        except Exception:
            _fallback = _DEFAULT_VALUE_POOLS.get(_key, '0, 1')
            st.warning(f'Could not parse `{_key}`; using default `{_fallback}`')
            return np.asarray([float(x) for x in _fallback.split(',')])

    def _parse_range(_key, _range_str):
        try:
            _parts = [float(x) for x in _range_str.split(',') if x.strip()]
            if len(_parts) != 2:
                raise ValueError
            _lo, _hi = sorted(_parts)
            if _lo == _hi:
                raise ValueError
            return _lo, _hi
        except Exception:
            _fallback = _DEFAULT_RANGES.get(_key, '-1, 1')
            st.warning(f'Could not parse `{_key}` range; using default `{_fallback}`')
            _parts = [float(x) for x in _fallback.split(',')]
            return sorted(_parts)

    _rec_value_pools = {}
    for _key in _rec_input_keys:
        _info = (_spec.get('inputs') or {}).get(_key, {})
        _descr = _info.get('description', 'trial input regressor') if _info else None

        if _is_logistic:
            col_mode, col_vals = st.columns([1, 2])
            with col_mode:
                _mode_choice = st.radio(
                    f'`{_key}` sampling',
                    ['Discrete', 'Continuous'],
                    key=f'rec_input_mode_{_rec_bundle["family"]}_{_key}',
                    help=('Discrete: sample uniformly from a list of values. '
                          'Continuous: sample uniformly from a range.'),
                )
            with col_vals:
                if _mode_choice == 'Continuous':
                    _range_str = st.text_input(
                        f'`{_key}` range (min, max)',
                        value=_DEFAULT_RANGES.get(_key, '-1, 1'),
                        key=f'rec_input_{_rec_bundle["family"]}_{_key}_range',
                        help='Two comma-separated numbers: lower and upper bound.',
                    )
                    _lo, _hi = _parse_range(_key, _range_str)
                    _rec_value_pools[_key] = {
                        'mode': 'continuous',
                        'range': (float(_lo), float(_hi)),
                    }
                else:
                    _vals_str = st.text_input(
                        f'`{_key}` values',
                        value=_DEFAULT_VALUE_POOLS.get(_key, '0, 1'),
                        key=f'rec_input_{_rec_bundle["family"]}_{_key}',
                        help='Comma-separated list of values; sampled uniformly per trial.',
                    )
                    _rec_value_pools[_key] = {
                        'mode': 'discrete',
                        'values': _parse_discrete(_key, _vals_str),
                    }
        else:
            _label = (
                f'`{_key}` — {_descr}'
                if _info else f'`{_key}` values'
            )
            _vals_str = st.text_input(
                _label,
                value=_DEFAULT_VALUE_POOLS.get(_key, '0, 1'),
                key=f'rec_input_{_rec_bundle["family"]}_{_key}',
                help='Comma-separated list of values; sampled uniformly per trial.',
            )
            _rec_value_pools[_key] = {
                'mode': 'discrete',
                'values': _parse_discrete(_key, _vals_str),
            }

    st.divider()

    # ------------------------------------------------------------------
    # 3. Trajectory shape sliders + live mini-plot per parameter
    # ------------------------------------------------------------------
    st.subheader('3. True parameter trajectories')

    _rec_traj_method = st.radio(
        'Trajectory generation method',
        ['Slider-driven sinusoid', 'REINFORCE forward simulation'],
        horizontal=True,
        key='rec_traj_method',
        help=(
            '**Slider-driven** — each trajectory is a closed-form sinusoid '
            'set by offset, slope, amplitude, period, and phase sliders. '
            '**REINFORCE** — start from a fixed point and let each trajectory '
            'evolve under the policy-gradient learning rule (Williams 1992) '
            'plus Gaussian random walk. Gives a realistic "true" trajectory '
            'for testing whether the EB fit recovers it both with and '
            'without the matching learning-rule prior.'
        ),
    )
    _rec_use_reinforce = (_rec_traj_method == 'REINFORCE forward simulation')

    if _rec_use_reinforce:
        st.caption(
            'Trial *t*: sample a response, compute reward (1 if choice '
            'matches `sign(c)`, else 0; tie-broken at random when `c = 0`), '
            'update each parameter by `α · ∇_θ log p(y | x, θ)` × reward, '
            'then add `𝒩(0, σ²_walk)` noise.'
        )
    else:
        st.caption(
            'Each trajectory is `offset + slope · (t / N) + amplitude · sin(2π t / period + phase)`. '
            'Set amplitude = 0 for a flat-or-linear evolution; sweep the phase '
            'slider to shift where the oscillation starts. The mini-plot in each '
            'expander updates live as you move the sliders.'
        )

    _default_E0_fn = _rec_bundle.get('default_E0')
    if callable(_default_E0_fn):
        try:
            _default_E0_arr = np.asarray(_default_E0_fn(N_rec), dtype=float)
        except Exception:
            _default_E0_arr = None
    else:
        _default_E0_arr = None

    _RACE_TRAJ_DEFAULTS = {
        'wr': dict(offset=1.5, amplitude=0.20, period_frac=0.25, slope=0.5),
        'wl': dict(offset=1.5, amplitude=0.20, period_frac=0.25, slope=0.5),
        'br': dict(offset=0.5, amplitude=0.05, period_frac=0.20, slope=0.0),
        'bl': dict(offset=0.5, amplitude=0.05, period_frac=0.20, slope=0.0),
        'z':  dict(offset=1.0, amplitude=0.05, period_frac=0.40, slope=-0.2),
    }

    # Per-family parameter bounds.  For race / DDM these are hard mathematical
    # constraints (e.g. DDM `a > 0`, race accumulator drifts > 0) — the slider
    # range and trajectory clipping both enforce them so the simulator/fit
    # never sees an invalid parameter.
    #
    # For logistic regression, weights and bias are mathematically unbounded;
    # the entries below are only the *default* slider range and the user can
    # widen them per parameter in section 3 (no clipping is applied).
    _PARAM_BOUNDS = {
        'race': {
            'wr': (0.05, 5.0), 'wl': (0.05, 5.0),
            'br': (0.05, 3.0), 'bl': (0.05, 3.0),
            'z':  (0.10, 5.0),
        },
        'ddm_exact': {
            'w': (-3.0, 5.0), 'b': (-2.0, 2.0),
            'a': (0.20, 4.0),         # boundary separation must be > 0
        },
        'logistic': {
            'w': (-5.0, 10.0), 'b': (-3.0, 3.0),
        },
    }

    # Logistic weights have no mathematical constraints, so we don't clip the
    # trajectory and we let the user widen the slider range freely.
    _CLIP_TRAJECTORY = _rec_bundle['family'] != 'logistic'

    def _bounds_for(name):
        """Default (lo, hi) slider range for the named parameter.

        For race / DDM these are also the trajectory-clipping bounds; for
        logistic they are only a UI default that the user can override.
        """
        family_bounds = _PARAM_BOUNDS.get(_rec_bundle['family'], {})
        if name in family_bounds:
            return family_bounds[name]
        # Generic logistic-style weight: any param named 'w' or 'w_<feature>'
        # uses the same default range as the canonical weight in this family.
        if _rec_bundle['family'] == 'logistic' and (name == 'w' or name.startswith('w_')):
            return family_bounds.get('w', (-5.0, 10.0))
        return (-5.0, 5.0)

    def _default_traj(k, name):
        # Returns (offset, amplitude, period, slope, phase). Phase is always 0
        # so the existing defaults reproduce the previous trajectories.
        if _rec_bundle['family'] == 'race' and name in _RACE_TRAJ_DEFAULTS:
            d = _RACE_TRAJ_DEFAULTS[name]
            return (d['offset'], d['amplitude'],
                    int(d['period_frac'] * N_rec), d['slope'], 0.0)
        if _default_E0_arr is not None and k < _default_E0_arr.shape[0]:
            row = _default_E0_arr[k]
            return (float(row[0]), 0.0, max(20, N_rec // 4),
                    float(row[-1] - row[0]), 0.0)
        return 0.0, 0.0, max(20, N_rec // 4), 0.0, 0.0

    def _make_traj(offset, amplitude, period, slope, phase, N, bounds=None):
        """Build the slider-driven trajectory and clip to (lo, hi) if bounds are set.

        The clip means amplitude/slope can be set freely without the resulting
        trajectory ever leaving the model's valid parameter region — useful
        when the user wants to explore extreme values without crashing the fit.

        ``phase`` is in radians and shifts the sinusoid: ``sin(2π t / period + phase)``.
        """
        t = np.arange(N)
        traj = (offset + slope * (t / N)
                + amplitude * np.sin(2 * np.pi * t / max(period, 1) + phase))
        if bounds is not None:
            lo, hi = bounds
            traj = np.clip(traj, lo, hi)
        return traj

    _rec_traj_specs = {}
    # REINFORCE settings (only filled when _rec_use_reinforce is True);
    # populated below the slider block.
    _rec_reinforce_cfg = None

    if _rec_use_reinforce:
        st.markdown(
            '**Initial parameter values, learning rates, and random-walk '
            'noise** — one row per parameter.  Defaults reflect the model\'s '
            '`default_E0`/`default_hyper` where available.'
        )
        _re_cfg = {'params_0': {}, 'alpha': {}, 'sigma_walk': {}}
        # Reasonable per-family alpha defaults: weights learn ~10× faster
        # than baselines for the race model; threshold is mostly stable.
        _ALPHA_DEFAULTS = {
            'race': {'wr': 0.05, 'wl': 0.05, 'br': 0.02, 'bl': 0.02, 'z': 0.0},
            'ddm_exact': {'w': 0.05, 'b': 0.02, 'a': 0.0},
            'logistic': None,  # all params get 0.05
        }
        family_alphas = _ALPHA_DEFAULTS.get(_rec_bundle['family'], None)
        for k, name in enumerate(_rec_bundle['PARAM_NAMES']):
            # Default initial value: first column of default_E0 if available,
            # else 0 (logistic) or 0.5 (other).
            if _default_E0_arr is not None and k < _default_E0_arr.shape[0]:
                d_init = float(_default_E0_arr[k, 0])
            else:
                d_init = 0.0
            d_alpha = (family_alphas or {}).get(name, 0.05)
            d_sigma = 0.02

            cols = st.columns(3)
            with cols[0]:
                _re_cfg['params_0'][name] = st.number_input(
                    f'`{name}` initial value',
                    value=float(d_init),
                    step=0.05,
                    format='%.4f',
                    key=f'rec_re_init_{_rec_bundle["family"]}_{name}',
                )
            with cols[1]:
                _re_cfg['alpha'][name] = st.number_input(
                    f'`{name}` learning rate α',
                    value=float(d_alpha),
                    min_value=0.0,
                    step=0.005,
                    format='%.4f',
                    key=f'rec_re_alpha_{_rec_bundle["family"]}_{name}',
                    help='Set to 0 to freeze this parameter at its initial '
                         'value (only the random walk will move it).',
                )
            with cols[2]:
                _re_cfg['sigma_walk'][name] = st.number_input(
                    f'`{name}` σ_walk',
                    value=float(d_sigma),
                    min_value=0.0,
                    step=0.005,
                    format='%.4f',
                    key=f'rec_re_sigma_{_rec_bundle["family"]}_{name}',
                    help='Per-parameter Gaussian random-walk std added on '
                         'top of the REINFORCE update each trial.',
                )
        _rec_reinforce_cfg = _re_cfg

    # The big per-parameter slider expanders only run in slider-driven mode;
    # in REINFORCE mode the trajectories come from the forward simulator.
    _slider_loop = (enumerate(_rec_bundle['PARAM_NAMES'])
                    if not _rec_use_reinforce else iter([]))
    for k, name in _slider_loop:
        with st.expander(f'`{name}` trajectory shape', expanded=(k == 0)):
            _default_lo, _default_hi = _bounds_for(name)

            # For logistic, expose the slider range as user-defined since the
            # parameters are mathematically unbounded.
            if not _CLIP_TRAJECTORY:
                rcol1, rcol2 = st.columns(2)
                with rcol1:
                    lo = float(st.number_input(
                        f'{name}: slider min',
                        value=float(_default_lo),
                        step=0.5,
                        key=f'rec_{_rec_bundle["family"]}_{name}_lo',
                        help='Lower end of the slider range. Widen freely — '
                             'logistic weights/bias are unbounded.',
                    ))
                with rcol2:
                    hi = float(st.number_input(
                        f'{name}: slider max',
                        value=float(_default_hi),
                        step=0.5,
                        key=f'rec_{_rec_bundle["family"]}_{name}_hi',
                        help='Upper end of the slider range.',
                    ))
                if hi <= lo:
                    st.warning(
                        f'`{name}`: slider max must exceed min — using defaults.'
                    )
                    lo, hi = float(_default_lo), float(_default_hi)
            else:
                lo, hi = float(_default_lo), float(_default_hi)

            d_offset, d_amp, d_period, d_slope, d_phase = _default_traj(k, name)
            d_offset = float(np.clip(d_offset, lo, hi))
            range_span = hi - lo

            # Slider step: pick something reasonable for the range size.
            step_offset = max((hi - lo) / 200.0, 0.001)
            step_amp    = max(range_span / 200.0, 0.001)
            step_slope  = max(range_span / 200.0, 0.001)

            if _CLIP_TRAJECTORY:
                st.caption(
                    f'Valid range: **[{lo:g}, {hi:g}]**.  The trajectory is clipped to '
                    f'this range — sweep amplitude or slope past the bound to see '
                    f'the parameter saturate.'
                )
            else:
                st.caption(
                    f'Slider range: **[{lo:g}, {hi:g}]** — widen above if you want '
                    f'larger values. The trajectory is **not** clipped, so amplitude '
                    f'and slope can push it past these bounds.'
                )
            c1, c2 = st.columns(2)
            with c1:
                offset = st.slider(
                    f'{name}: offset', float(lo), float(hi),
                    d_offset, step_offset,
                    key=f'rec_{_rec_bundle["family"]}_{name}_offset',
                )
                amplitude = st.slider(
                    f'{name}: amplitude', 0.0, float(range_span),
                    float(min(d_amp, range_span)), step_amp,
                    key=f'rec_{_rec_bundle["family"]}_{name}_amp',
                )
            with c2:
                period = st.slider(
                    f'{name}: period (trials)',
                    min_value=20, max_value=max(40, N_rec),
                    value=int(min(d_period, max(20, N_rec))), step=10,
                    key=f'rec_{_rec_bundle["family"]}_{name}_period',
                )
                slope = st.slider(
                    f'{name}: linear slope (over all trials)',
                    -float(range_span), float(range_span),
                    float(np.clip(d_slope, -range_span, range_span)), step_slope,
                    key=f'rec_{_rec_bundle["family"]}_{name}_slope',
                )
                phase = st.slider(
                    f'{name}: phase (radians)',
                    -float(np.pi), float(np.pi),
                    float(d_phase), float(np.pi / 60),
                    key=f'rec_{_rec_bundle["family"]}_{name}_phase',
                    help='Shifts the sinusoid: 0 starts at the offset, '
                         'π/2 starts at the peak, π flips the cycle.',
                )
            _traj_bounds = (lo, hi) if _CLIP_TRAJECTORY else None
            _rec_traj_specs[name] = (
                float(offset), float(amplitude), int(period), float(slope),
                float(phase), _traj_bounds,
            )

            # Live preview of this trajectory (post-clip if applicable, so
            # the user sees exactly what the simulator will get).
            _traj_k = _make_traj(offset, amplitude, period, slope, phase,
                                 N_rec, bounds=_traj_bounds)
            _mini_fig, _mini_ax = plt.subplots(figsize=(5.5, 1.6))
            _style_fig(_mini_fig)
            _style_ax(_mini_ax, xlabel='Trial', title=f'{name} preview')
            _mini_ax.plot(np.arange(N_rec), _traj_k, color='#000000', lw=1.0)
            _mini_ax.axhline(lo, color='#cc4444', lw=0.6, ls=':', alpha=0.6)
            _mini_ax.axhline(hi, color='#cc4444', lw=0.6, ls=':', alpha=0.6)
            _mini_fig.tight_layout()
            st.pyplot(_mini_fig, use_container_width=True)
            plt.close(_mini_fig)

    # Build the truth trajectory matrix when in slider-driven mode.  In
    # REINFORCE mode the trajectory is produced by the forward simulator
    # inside the worker thread (it depends on the random seed and reward
    # outcomes), so true_params is built there instead.
    if _rec_use_reinforce:
        true_params = None
    else:
        true_params = np.stack([
            _make_traj(*_rec_traj_specs[name][:5], N_rec,
                       bounds=_rec_traj_specs[name][5])
            for name in _rec_bundle['PARAM_NAMES']
        ])

    st.divider()

    # ------------------------------------------------------------------
    # 4. Model-level hyperparameters (true vs init)
    # ------------------------------------------------------------------
    _default_mh_fn = _rec_bundle.get('default_model_hyper') or (lambda: {})
    try:
        _default_mh = dict(_default_mh_fn() or {})
    except Exception:
        _default_mh = {}
    _rec_true_mh, _rec_init_mh = {}, {}
    if _default_mh:
        st.subheader('4. Model-level hyperparameters')
        st.caption(
            'These are constants the model exposes to Empirical Bayes (e.g. the '
            'race model\'s within-trial accumulator noise `sig_i`). The simulator '
            'uses the **true** value; the EB outer loop is started from the **init** '
            'value and reports a recovered estimate in the result.'
        )
        for _key, _val in _default_mh.items():
            _max = max(2.0, 4.0 * float(_val))
            col_t, col_i = st.columns(2)
            with col_t:
                _rec_true_mh[_key] = st.slider(
                    f'True `{_key}` (simulator)',
                    min_value=0.001, max_value=float(_max),
                    value=float(_val), step=0.001,
                    key=f'rec_true_mh_{_rec_bundle["family"]}_{_key}',
                )
            with col_i:
                _rec_init_mh[_key] = st.slider(
                    f'Initial `{_key}` (EB starting point)',
                    min_value=0.001, max_value=float(_max),
                    value=float(_val), step=0.001,
                    key=f'rec_init_mh_{_rec_bundle["family"]}_{_key}',
                )
        st.divider()

    # ------------------------------------------------------------------
    # 5. Run recovery (threaded, verbose)
    # ------------------------------------------------------------------
    st.subheader(f'{"5" if _default_mh else "4"}. Run recovery')
    st.info(
        'Model-recovery fits run inside the Streamlit session and may be '
        'slower than the same code on your local machine. Treat this page as '
        'an interactive sanity check for the recovery workflow.',
    )

    # Session-state setup
    if 'rec_running' not in st.session_state:
        st.session_state['rec_running'] = False
    if 'rec_result' not in st.session_state:
        st.session_state['rec_result'] = None
    if 'rec_log' not in st.session_state:
        st.session_state['rec_log'] = []
    if 'rec_error' not in st.session_state:
        st.session_state['rec_error'] = None

    # Dual-fit comparison: only meaningful when truth came from REINFORCE.
    if _rec_use_reinforce:
        _rec_compare_fits = st.checkbox(
            'Also fit with zero-centered prior (compare log evidence)',
            value=True,
            key='rec_compare_fits',
            help=(
                'Runs two fits on the same simulated data: '
                '(a) **REINFORCE prior** — random-walk transition mean is '
                'the policy-gradient score function (matches the truth-'
                'generating process), and '
                '(b) **zero-centred prior** — vanilla random walk with '
                'mean 0 (no learning rule). '
                'Compares log_evidence to quantify how much the matched '
                'prior helps.'
            ),
        )
    else:
        _rec_compare_fits = False

    run_rec = st.button(
        'Simulate + fit', key='rec_run_btn', type='primary',
        disabled=st.session_state['rec_running'],
    )

    if run_rec:
        st.session_state['rec_running'] = True
        st.session_state['rec_result']  = None
        st.session_state['rec_log']     = []
        st.session_state['rec_error']   = None

        _rec_q = queue.Queue()

        class _RecQueueTqdm:
            def __init__(self, *a, **kw):
                self._n = 0; self._map_n = 0; self._postfix = {}
            def update(self, n=1):
                self._n += n; self._map_n = 0
                self._postfix.pop('MAP loss', None)
                _rec_q.put(('progress', self._n, self._map_n, dict(self._postfix)))
            def set_postfix(self, d, **kwargs):
                self._postfix.update(d)
                if 'MAP loss' in d:
                    self._map_n += 1
                _rec_q.put(('progress', self._n, self._map_n, dict(self._postfix)))
            def close(self): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass

        def _rec_status_cb(payload):
            _rec_q.put(('status', payload))

        _rec_orig_tqdm = _rec_hyper_opt_mod.tqdm
        _rec_hyper_opt_mod.tqdm = _RecQueueTqdm

        # Snapshot inputs into locals for the worker thread
        _bundle_local        = _rec_bundle
        _N_local             = int(N_rec)
        _seed_local          = int(seed_rec)
        _true_params_local   = (np.array(true_params, copy=True)
                                if true_params is not None else None)
        _value_pools_local   = {
            k: (
                {'mode': 'continuous', 'range': tuple(spec['range'])}
                if spec.get('mode') == 'continuous'
                else {'mode': 'discrete',
                      'values': np.array(spec['values'], copy=True)}
            )
            for k, spec in _rec_value_pools.items()
        }
        _true_mh_local       = dict(_rec_true_mh)
        _init_mh_local       = dict(_rec_init_mh)
        _use_reinforce_local = bool(_rec_use_reinforce)
        _compare_fits_local  = bool(_rec_compare_fits)
        _reinforce_cfg_local = (
            None if not _use_reinforce_local else {
                'params_0':   np.array(
                    [_rec_reinforce_cfg['params_0'][n]
                     for n in _rec_bundle['PARAM_NAMES']], dtype=float),
                'alpha':      np.array(
                    [_rec_reinforce_cfg['alpha'][n]
                     for n in _rec_bundle['PARAM_NAMES']], dtype=float),
                'sigma_walk': np.array(
                    [_rec_reinforce_cfg['sigma_walk'][n]
                     for n in _rec_bundle['PARAM_NAMES']], dtype=float),
            }
        )

        def _run_recovery():
            try:
                rng = np.random.default_rng(_seed_local)
                inputs = {}
                for k, spec in _value_pools_local.items():
                    if spec.get('mode') == 'continuous':
                        lo, hi = spec['range']
                        inputs[k] = rng.uniform(lo, hi, size=_N_local)
                    else:
                        inputs[k] = rng.choice(spec['values'], size=_N_local)

                t0 = _rec_time.time()
                if _use_reinforce_local:
                    _rec_status_cb({
                        'message': f'Forward-simulating {_N_local} REINFORCE trials…',
                        'stage':   'simulate',
                    })
                    from psytrax.learning_rules import (
                        make_reinforce, simulate_with_learning_rule,
                    )
                    _lr = make_reinforce(_bundle_local['log_lik_trial'])
                    _true_params_run, data = simulate_with_learning_rule(
                        _bundle_local['sample_trial'], _lr,
                        params_0=_reinforce_cfg_local['params_0'],
                        inputs=inputs,
                        alpha=_reinforce_cfg_local['alpha'],
                        sigma_walk=_reinforce_cfg_local['sigma_walk'],
                        rng=rng, model_hyper=_true_mh_local,
                    )
                else:
                    _rec_status_cb({'message': f'Simulating {_N_local} trials…',
                                    'stage': 'simulate'})
                    data = psytrax.simulate(
                        _bundle_local['sample_trial'],
                        _true_params_local,
                        inputs,
                        rng=rng,
                        model_hyper=_true_mh_local,
                    )
                    _true_params_run = _true_params_local
                t_sim = _rec_time.time() - t0
                _rec_status_cb({'message': f'Simulated in {t_sim:.1f}s — running EB fit…',
                                'stage': 'fit_start'})

                fit_kwargs = dict(
                    data=data,
                    log_lik_trial=_bundle_local['log_lik_trial'],
                    n_params=_bundle_local['N_PARAMS'],
                    param_names=_bundle_local['PARAM_NAMES'],
                    hess_calc='All',   # weights + hyperparameter CIs
                    verbose=True,
                    status_callback=_rec_status_cb,
                )
                _dh = _bundle_local.get('default_hyper')
                if callable(_dh):
                    try:
                        fit_kwargs['hyper'] = _dh()
                    except Exception:
                        pass
                _de = _bundle_local.get('default_E0')
                if callable(_de):
                    try:
                        fit_kwargs['E0'] = _de(_N_local)
                    except Exception:
                        pass
                if _init_mh_local:
                    fit_kwargs['model_hyper'] = _init_mh_local

                # The "primary" fit always uses the same prior the truth
                # was generated under: REINFORCE in REINFORCE mode, zero-
                # centred otherwise.  When the user has asked for the
                # comparison, we run a SECOND fit afterwards with the
                # opposite prior (zero-centred) on the same data.
                primary_kwargs = dict(fit_kwargs)
                if _use_reinforce_local:
                    from psytrax.learning_rules import make_reinforce
                    primary_kwargs['learning_rule'] = make_reinforce(
                        _bundle_local['log_lik_trial']
                    )
                    primary_label = 'REINFORCE prior'
                else:
                    primary_label = 'zero-centred prior'

                t0 = _rec_time.time()
                result = psytrax.fit(**primary_kwargs)
                t_fit = _rec_time.time() - t0
                result['fit_label']        = primary_label

                # Optional companion fit with the opposite prior.
                companion = None
                if _compare_fits_local and _use_reinforce_local:
                    _rec_status_cb({
                        'message': ('Comparison fit with zero-centred prior '
                                    '(no learning rule)…'),
                        'stage':   'fit_compare',
                    })
                    comp_kwargs = dict(fit_kwargs)
                    comp_kwargs.pop('learning_rule', None)
                    t0c = _rec_time.time()
                    companion = psytrax.fit(**comp_kwargs)
                    companion['fit_label']    = 'zero-centred prior'
                    companion['fit_time']     = _rec_time.time() - t0c
                    companion['true_params']  = _true_params_run
                    companion['true_model_hyper'] = _true_mh_local

                # Augment the result with truth so it's self-contained
                result['true_params']      = _true_params_run
                result['true_model_hyper'] = _true_mh_local
                result['simulated_data']   = data
                result['simulate_time']    = t_sim
                result['fit_time']         = t_fit
                result['model_family']     = _bundle_local['family']
                result['traj_method']      = (
                    'reinforce' if _use_reinforce_local else 'slider')
                if _use_reinforce_local:
                    result['reinforce_cfg'] = _reinforce_cfg_local
                if companion is not None:
                    result['companion_fit'] = companion
                result['input_modes']      = {
                    k: spec.get('mode', 'discrete')
                    for k, spec in _value_pools_local.items()
                }

                _rec_q.put(('done', result))
            except Exception:
                import traceback
                _rec_q.put(('error', traceback.format_exc()))
            finally:
                _rec_hyper_opt_mod.tqdm = _rec_orig_tqdm

        _t = threading.Thread(target=_run_recovery, daemon=True)
        _t.start()
        st.session_state['_rec_thread'] = _t
        st.session_state['_rec_queue']  = _rec_q

    # Stream progress while the thread is alive ----------------------
    if st.session_state['rec_running']:
        _rec_q  = st.session_state['_rec_queue']
        _t      = st.session_state['_rec_thread']

        st.markdown('**Recovery in progress…** &nbsp; `simulate → fit`')
        col_cyc, col_map = st.columns(2)
        cycle_text   = col_cyc.empty()
        map_text     = col_map.empty()
        status_text  = st.empty()
        log_evd_text = st.empty()
        log_box      = st.empty()

        cycle, map_iter = 0, 0
        log_evd_str, best_str, map_loss_str = '—', '—', '—'
        current_status = 'Preparing…'
        rec_log = st.session_state.get('rec_log', [])
        terminal = None

        while _t.is_alive():
            while not _rec_q.empty():
                try:
                    msg = _rec_q.get_nowait()
                    if msg[0] == 'progress':
                        _, cycle, map_iter, postfix = msg
                        log_evd_str = postfix.get('log_evd', log_evd_str)
                        best_str = postfix.get('best', best_str)
                        map_loss_str = postfix.get('MAP loss', map_loss_str)
                    elif msg[0] == 'status':
                        payload = msg[1]
                        current_status = payload.get('message', current_status)
                        rec_log.append(current_status)
                        rec_log = rec_log[-12:]
                        st.session_state['rec_log'] = rec_log
                    elif msg[0] in ('done', 'error'):
                        terminal = msg
                except queue.Empty:
                    break
            cycle_text.metric('Cycles completed', cycle)
            map_text.metric('MAP iters (current cycle)', map_iter)
            status_text.markdown(f'**Current step:** {current_status}')
            log_evd_text.markdown(
                f'Log evidence (higher is better) — current: **{log_evd_str}** &nbsp;|&nbsp; '
                f'best: **{best_str}**'
                + (f' &nbsp;|&nbsp; Neg log-posterior (lower is better): **{map_loss_str}**'
                   if map_loss_str != '—' else '')
            )
            if rec_log:
                log_box.code('\n'.join(rec_log), language='text')
            _rec_time.sleep(0.5)

        while not _rec_q.empty():
            try:
                msg = _rec_q.get_nowait()
                if msg[0] == 'progress':
                    _, cycle, map_iter, postfix = msg
                    log_evd_str = postfix.get('log_evd', log_evd_str)
                    best_str = postfix.get('best', best_str)
                    map_loss_str = postfix.get('MAP loss', map_loss_str)
                elif msg[0] == 'status':
                    payload = msg[1]
                    current_status = payload.get('message', current_status)
                    rec_log.append(current_status)
                    rec_log = rec_log[-12:]
                elif msg[0] in ('done', 'error'):
                    terminal = msg
            except queue.Empty:
                break

        st.session_state['rec_log'] = rec_log
        st.session_state['rec_running'] = False
        if terminal is None:
            terminal = ('error', 'No result received from recovery thread.')
        msg_type, payload = terminal[0], terminal[1]
        if msg_type == 'done':
            st.session_state['rec_result'] = payload
            st.session_state['rec_error'] = None
        else:
            st.session_state['rec_error'] = payload
            st.error(f'Recovery failed:\n```\n{payload}\n```')
        st.rerun()

    # --- Download button (visible as soon as a fit has finished) -------
    _rec_result_for_dl = st.session_state.get('rec_result')
    if _rec_result_for_dl is not None:
        _dl_buf = io.BytesIO()
        np.save(_dl_buf, _rec_result_for_dl, allow_pickle=True)
        _dl_buf.seek(0)
        st.download_button(
            'Download recovery result (.npy)',
            data=_dl_buf.getvalue(),
            file_name=f'{_rec_bundle["family"]}_model_recovery.npy',
            mime='application/octet-stream',
            help='Pickled dict containing the recovered fit, the truth '
                 'trajectories, and the simulated data used for the fit.',
            key='rec_download_top',
        )

    # ------------------------------------------------------------------
    # 6. Results
    # ------------------------------------------------------------------
    result_rec = st.session_state.get('rec_result')
    if result_rec is not None:
        _rec_done_msg = (
            f'Done — simulated **{result_rec["params"].shape[1]}** trials in '
            f'{result_rec.get("simulate_time", 0):.1f}s, '
            f'fit in {result_rec.get("fit_time", 0):.1f}s.  '
            f'Log evidence: {result_rec["log_evidence"]:.2f}.'
        )
        st.success(_rec_done_msg)

        # --- Recovered model_hyper -------------------------------------
        rec_mh  = result_rec.get('model_hyper') or {}
        true_mh = result_rec.get('true_model_hyper') or {}
        if rec_mh or true_mh:
            st.subheader('Recovered model_hyper')
            mh_keys = sorted(set(rec_mh) | set(true_mh))
            cols = st.columns(min(3, max(1, len(mh_keys))))
            for i, key in enumerate(mh_keys):
                tval = true_mh.get(key, np.nan)
                rval = rec_mh.get(key, np.nan)
                with cols[i % len(cols)]:
                    delta = (rval - tval) if (np.isfinite(tval) and np.isfinite(rval)) else None
                    st.metric(
                        f'{key}',
                        f'true {tval:.4f}  →  rec {rval:.4f}',
                        delta=f'{delta:+.4f}' if delta is not None else None,
                    )

        # --- Trajectory overlay ---------------------------------------
        recovered     = result_rec['params']
        true_params_r = result_rec['true_params']
        param_names_r = result_rec['param_names']
        K_r, N_r      = recovered.shape
        W_std_rec     = (result_rec.get('hess_info') or {}).get('W_std')
        companion_fit = result_rec.get('companion_fit')

        n_cols = min(K_r, 3)
        n_rows = int(np.ceil(K_r / n_cols))
        fig_rec, axes_rec = plt.subplots(n_rows, n_cols,
                                         figsize=(5 * n_cols, 3 * n_rows),
                                         squeeze=False)
        _tc = _style_fig(fig_rec)
        trials_rec = np.arange(N_r)

        primary_label   = result_rec.get('fit_label',   'recovered')
        companion_label = (companion_fit.get('fit_label', 'recovered (alt prior)')
                           if companion_fit else None)
        recovered_alt   = companion_fit['params'] if companion_fit else None
        W_std_alt       = ((companion_fit.get('hess_info') or {}).get('W_std')
                           if companion_fit else None)

        per_param_summary = []
        for k, (ax, name) in enumerate(zip(axes_rec.flat, param_names_r)):
            _style_ax(ax, xlabel='Trial', title=name)
            ax.plot(trials_rec, true_params_r[k], color='#000000', lw=1.5, label='true')
            ax.plot(trials_rec, recovered[k], color='#4e9af1', lw=1.0,
                    label=primary_label)
            if W_std_rec is not None:
                ax.fill_between(
                    trials_rec,
                    recovered[k] - W_std_rec[k],
                    recovered[k] + W_std_rec[k],
                    color='#4e9af1', alpha=0.15,
                )
            if recovered_alt is not None:
                ax.plot(trials_rec, recovered_alt[k], color='#f1a44e', lw=1.0,
                        ls='--', label=companion_label)
                if W_std_alt is not None:
                    ax.fill_between(
                        trials_rec,
                        recovered_alt[k] - W_std_alt[k],
                        recovered_alt[k] + W_std_alt[k],
                        color='#f1a44e', alpha=0.10,
                    )
            if k == 0:
                _style_legend(ax)
            mae = float(np.mean(np.abs(recovered[k] - true_params_r[k])))
            if np.std(true_params_r[k]) > 0 and np.std(recovered[k]) > 0:
                corr = float(np.corrcoef(recovered[k], true_params_r[k])[0, 1])
            else:
                corr = float('nan')
            row = {'parameter': name,
                   f'MAE ({primary_label})': mae,
                   f'corr ({primary_label})': corr}
            if recovered_alt is not None:
                mae_a = float(np.mean(np.abs(recovered_alt[k] - true_params_r[k])))
                if np.std(true_params_r[k]) > 0 and np.std(recovered_alt[k]) > 0:
                    corr_a = float(np.corrcoef(recovered_alt[k], true_params_r[k])[0, 1])
                else:
                    corr_a = float('nan')
                row[f'MAE ({companion_label})']  = mae_a
                row[f'corr ({companion_label})'] = corr_a
            per_param_summary.append(row)

        for ax in axes_rec.flat[K_r:]:
            ax.set_visible(False)
        fig_rec.suptitle('Parameter recovery: true vs recovered',
                         color=_tc['text'], fontsize=12)
        fig_rec.tight_layout(rect=[0, 0, 1, 0.96], pad=1.3)
        _show_fig(fig_rec, 'recovery_overlay.png')

        # Log-evidence comparison block (only when both fits ran).
        if companion_fit is not None:
            le_p = float(result_rec.get('log_evidence', float('nan')))
            le_c = float(companion_fit.get('log_evidence', float('nan')))
            d    = le_p - le_c
            st.markdown('**Log-evidence comparison**')
            cmp_cols = st.columns(3)
            cmp_cols[0].metric(primary_label,   f'{le_p:.2f}')
            cmp_cols[1].metric(companion_label, f'{le_c:.2f}')
            cmp_cols[2].metric('Δ (primary − alt)', f'{d:+.2f}',
                               help=('Positive Δ means the matched-prior '
                                     '(REINFORCE) fit assigns higher marginal '
                                     'likelihood to the simulated data than '
                                     'the zero-centred prior.'))

        st.subheader('Per-parameter recovery quality')
        st.dataframe(
            pd.DataFrame(per_param_summary).set_index('parameter').round(4),
            use_container_width=True,
        )

        # --- Behavioural recovery: 4-quartile psychometric & chronometric ----
        st.subheader('Behavioural recovery: truth vs recovered, by quartile')
        st.caption(
            'Trials are split into four equally-sized windows (early → late). '
            'For each window we plot the empirical bins from the simulator (black '
            'dots) and from a fresh re-simulation of the recovered trajectory '
            '(blue x\'s), and — for built-in models — overlay the analytic '
            'psychometric (and chronometric) curves computed from the parameter '
            'values in that window: black for truth, blue dashed for recovered. '
            'Close agreement means the recovered trajectory reproduces the '
            'simulator\'s behaviour even when individual parameters were poorly '
            'identified.'
        )

        sim_data_truth = result_rec.get('simulated_data') or {}
        inputs_truth   = sim_data_truth.get('inputs') or {}

        # Detect the model family / fixed-params for analytic-curve helpers.
        # We need to swap in the *true* model_hyper for truth-side curves and
        # the *recovered* model_hyper for recovered-side curves.
        model_family, _ = _model_family_info(param_names_r, result_rec)

        truth_fixed_params = dict(true_mh)   # e.g. {'sig_i': true value}
        rec_fixed_params   = dict(rec_mh)    # e.g. {'sig_i': recovered value}

        # The behavioural plots assume signed contrast `c` is the (only) thing
        # driving choice — empirical psychometric bins use it as the x-axis,
        # and the logistic analytic curve treats every other regressor as 0.
        # For race / DDM that's always true; for logistic it's only meaningful
        # when the user fits a single discrete `c` regressor.
        _input_modes = result_rec.get('input_modes') or {}
        _input_keys_truth = list(inputs_truth.keys())
        if model_family == 'logistic':
            _show_behavioural = (
                _input_keys_truth == ['c']
                and _input_modes.get('c', 'discrete') == 'discrete'
            )
            _skip_reason = (
                'Behavioural curves are only shown for logistic models with a '
                'single discrete `c` regressor — multi-regressor or continuous '
                'inputs make the psychometric x-axis ambiguous, so the section '
                'is omitted.'
            )
        else:
            _show_behavioural = 'c' in inputs_truth
            _skip_reason = (
                'Behavioural curves require a `c` input. The selected model '
                'doesn\'t expose one — skipping psychometric/chronometric plots.'
            )

        if _show_behavioural:
            # Resample button — each click bumps a counter that's mixed into
            # the RNG seed, so the empirical "recovered (sim)" markers shift
            # while the analytic curves and truth markers stay put. Useful
            # for eyeballing simulator-noise variability around the analytic
            # recovered curve.
            if 'rec_resample_count' not in st.session_state:
                st.session_state['rec_resample_count'] = 0
            _resample_col1, _resample_col2 = st.columns([1, 3])
            with _resample_col1:
                if st.button('Resample recovered',
                             key='rec_resample_button',
                             help='Draw a new simulation from the recovered '
                                  'trajectory to see how the blue crosses '
                                  'shift due to simulator noise alone.'):
                    st.session_state['rec_resample_count'] += 1
            with _resample_col2:
                _rcount = int(st.session_state['rec_resample_count'])
                st.caption(
                    f'Resample #{_rcount} '
                    f'(seed = {int(seed_rec) + 1 + _rcount}).'
                )
            try:
                _resample_seed = int(seed_rec) + 1 + int(
                    st.session_state['rec_resample_count']
                )
                rng_recover = np.random.default_rng(_resample_seed)
                data_rec_sim = psytrax.simulate(
                    _rec_bundle['sample_trial'],
                    recovered,
                    inputs_truth,
                    rng=rng_recover,
                    model_hyper=rec_mh,
                )

                c_data   = np.asarray(inputs_truth['c'])
                r_truth  = np.asarray(sim_data_truth['responses'])
                r_rec    = np.asarray(data_rec_sim['responses'])
                T_truth  = sim_data_truth.get('times')
                T_rec    = data_rec_sim.get('times')
                has_rt   = (T_truth is not None and T_rec is not None
                            and model_family in _RT_CURVE_FAMILIES)
                if T_truth is not None: T_truth = np.asarray(T_truth)
                if T_rec is not None:   T_rec   = np.asarray(T_rec)

                contrasts_unique = np.unique(c_data)
                c_grid = np.linspace(contrasts_unique.min(), contrasts_unique.max(), 100)

                N_WIN = 4
                edges = np.linspace(0, N_r, N_WIN + 1, dtype=int)

                # --- Psychometric quartiles ---------------------------------
                fig_p, axes_p = plt.subplots(2, 2, figsize=(11, 8))
                _tc = _style_fig(fig_p)
                for wi, ax in enumerate(axes_p.flat):
                    t0, t1 = int(edges[wi]), int(edges[wi + 1])
                    c_win = c_data[t0:t1]
                    r_t_win = r_truth[t0:t1]
                    r_r_win = r_rec[t0:t1]
                    c_uniq = np.unique(c_win)
                    p_t_emp = np.array([r_t_win[c_win == cv].mean() for cv in c_uniq])
                    p_r_emp = np.array([r_r_win[c_win == cv].mean() for cv in c_uniq])
                    n_w     = np.array([np.sum(c_win == cv) for cv in c_uniq])

                    _style_ax(ax, xlabel='Signed contrast', ylabel='P(right)',
                              title=f'Trials {t0 + 1}–{t1}')
                    sizes = [max(10, n / 4) for n in n_w]
                    ax.scatter(c_uniq, p_t_emp, s=sizes, color=_tc['text'],
                               alpha=0.85, label='truth (sim)')
                    ax.scatter(c_uniq, p_r_emp, s=sizes, color='#4e9af1',
                               alpha=0.85, marker='x', label='recovered (sim)')

                    # Analytic curves where available
                    p_t_curve, _ = _curve_predictions(
                        true_params_r[:, t0:t1], param_names_r, c_grid,
                        model_family, fixed_params=truth_fixed_params,
                    )
                    p_r_curve, _ = _curve_predictions(
                        recovered[:, t0:t1], param_names_r, c_grid,
                        model_family, fixed_params=rec_fixed_params,
                    )
                    if p_t_curve is not None:
                        ax.plot(c_grid, p_t_curve, color='#000000', lw=1.5,
                                label='truth (analytic)')
                    if p_r_curve is not None:
                        ax.plot(c_grid, p_r_curve, color='#4e9af1', lw=1.5,
                                ls='--', label='recovered (analytic)')

                    ax.axhline(0.5, color=_tc['text'], lw=0.5, ls='--', alpha=0.4)
                    ax.axvline(0,   color=_tc['text'], lw=0.5, ls='--', alpha=0.4)
                    ax.set_ylim(-0.05, 1.05)
                    if wi == 0:
                        _style_legend(ax, fontsize=8, loc='lower right')

                fig_p.suptitle('Psychometric quartiles: truth vs recovered',
                               color=_tc['text'], fontsize=12)
                fig_p.tight_layout()
                _show_fig(fig_p, 'recovery_psychometric_quartiles.png')

                # --- Chronometric quartiles (only for RT-bearing models) ----
                if has_rt:
                    with st.spinner('Computing chronometric quartiles…'):
                        fig_c, axes_c = plt.subplots(2, 2, figsize=(11, 8))
                        _tc = _style_fig(fig_c)
                        panel_data = []
                        y_series   = []

                        for wi in range(N_WIN):
                            t0, t1 = int(edges[wi]), int(edges[wi + 1])
                            c_win  = c_data[t0:t1]
                            T_t_w  = T_truth[t0:t1]
                            T_r_w  = T_rec[t0:t1]
                            c_uniq = np.unique(c_win)
                            rt_t_emp = np.array([T_t_w[c_win == cv].mean() for cv in c_uniq])
                            rt_r_emp = np.array([T_r_w[c_win == cv].mean() for cv in c_uniq])
                            n_w      = np.array([np.sum(c_win == cv) for cv in c_uniq])

                            _, rt_t_curve = _curve_predictions(
                                true_params_r[:, t0:t1], param_names_r, c_grid,
                                model_family, fixed_params=truth_fixed_params,
                            )
                            _, rt_r_curve = _curve_predictions(
                                recovered[:, t0:t1], param_names_r, c_grid,
                                model_family, fixed_params=rec_fixed_params,
                            )
                            panel_data.append((t0, t1, c_uniq, rt_t_emp, rt_r_emp,
                                               n_w, rt_t_curve, rt_r_curve))
                            y_series.extend([rt_t_emp, rt_r_emp,
                                             rt_t_curve, rt_r_curve])

                        shared_ylim = _shared_ylim(
                            [np.asarray(s) for s in y_series if s is not None]
                        )

                        for ax, (t0, t1, c_uniq, rt_t_emp, rt_r_emp,
                                n_w, rt_t_curve, rt_r_curve) in zip(
                                axes_c.flat, panel_data):
                            _style_ax(ax, xlabel='Signed contrast',
                                      ylabel='Mean RT (s)',
                                      title=f'Trials {t0 + 1}–{t1}')
                            sizes = [max(10, n / 4) for n in n_w]
                            ax.scatter(c_uniq, rt_t_emp, s=sizes, color=_tc['text'],
                                       alpha=0.85, label='truth (sim)')
                            ax.scatter(c_uniq, rt_r_emp, s=sizes, color='#4e9af1',
                                       alpha=0.85, marker='x',
                                       label='recovered (sim)')
                            if rt_t_curve is not None:
                                ax.plot(c_grid, rt_t_curve, color='#000000', lw=1.5,
                                        label='truth (analytic)')
                            if rt_r_curve is not None:
                                ax.plot(c_grid, rt_r_curve, color='#4e9af1', lw=1.5,
                                        ls='--', label='recovered (analytic)')
                            ax.axvline(0, color=_tc['text'], lw=0.5, ls='--', alpha=0.4)
                            if shared_ylim is not None:
                                ax.set_ylim(*shared_ylim)
                            if (t0, t1) == (panel_data[0][0], panel_data[0][1]):
                                _style_legend(ax, fontsize=8, loc='upper right')

                        fig_c.suptitle('Chronometric quartiles: truth vs recovered',
                                       color=_tc['text'], fontsize=12)
                        fig_c.tight_layout()
                        _show_fig(fig_c, 'recovery_chronometric_quartiles.png')

                # --- Dopamine quartiles (race + sig_DA only) ----------------
                _da_truth = sim_data_truth.get('dopamine')
                _da_rec   = data_rec_sim.get('dopamine') if data_rec_sim else None
                if (model_family == 'race'
                        and _da_truth is not None
                        and _da_rec is not None):
                    fig_d, axes_d = plt.subplots(2, 2, figsize=(11, 8))
                    _tc = _style_fig(fig_d)
                    da_truth = np.asarray(_da_truth, dtype=float)
                    da_rec   = np.asarray(_da_rec,   dtype=float)
                    try:
                        _wr_idx = list(param_names_r).index('wr')
                        _wl_idx = list(param_names_r).index('wl')
                    except ValueError:
                        _wr_idx = None

                    if _wr_idx is not None:
                        for wi, ax in enumerate(axes_d.flat):
                            t0, t1 = int(edges[wi]), int(edges[wi + 1])
                            c_win  = c_data[t0:t1]
                            da_t_w = da_truth[t0:t1]
                            da_r_w = da_rec[t0:t1]
                            mask_t = np.isfinite(da_t_w)
                            mask_r = np.isfinite(da_r_w)
                            c_uniq = np.unique(c_win)

                            da_t_emp = np.array([
                                da_t_w[mask_t & (c_win == cv)].mean()
                                if np.any(mask_t & (c_win == cv)) else np.nan
                                for cv in c_uniq
                            ])
                            da_r_emp = np.array([
                                da_r_w[mask_r & (c_win == cv)].mean()
                                if np.any(mask_r & (c_win == cv)) else np.nan
                                for cv in c_uniq
                            ])
                            n_w = np.array([np.sum(c_win == cv) for cv in c_uniq])

                            # Pull β and offset from each side's model_hyper
                            # (true_mh for the truth curve, rec_mh for the
                            # recovered curve), falling back to module-top
                            # defaults when missing.
                            _DA_BETA_T   = float(true_mh.get('da_beta',   _DA_BETA_DEFAULT))
                            _DA_OFFSET_T = float(true_mh.get('da_offset', _DA_OFFSET_DEFAULT))
                            _DA_BETA_R   = float(rec_mh.get('da_beta',    _DA_BETA_DEFAULT))
                            _DA_OFFSET_R = float(rec_mh.get('da_offset',  _DA_OFFSET_DEFAULT))
                            wr_t = true_params_r[_wr_idx, t0:t1]
                            wl_t = true_params_r[_wl_idx, t0:t1]
                            wr_r = recovered[_wr_idx, t0:t1]
                            wl_r = recovered[_wl_idx, t0:t1]
                            curve_t = np.array([
                                np.mean(_dopamine_tanh_readout(
                                    np.where(cv >= 0, wr_t, wl_t) * abs(cv),
                                    _DA_BETA_T,
                                    _DA_OFFSET_T,
                                ))
                                for cv in c_grid
                            ])
                            curve_r = np.array([
                                np.mean(_dopamine_tanh_readout(
                                    np.where(cv >= 0, wr_r, wl_r) * abs(cv),
                                    _DA_BETA_R,
                                    _DA_OFFSET_R,
                                ))
                                for cv in c_grid
                            ])

                            _style_ax(ax, xlabel='Signed contrast',
                                      ylabel='Dopamine peak (a.u.)',
                                      title=f'Trials {t0 + 1}–{t1}')
                            sizes = [max(10, n / 4) for n in n_w]
                            ax.scatter(c_uniq, da_t_emp, s=sizes,
                                       color=_tc['text'], alpha=0.85,
                                       label='truth (sim)')
                            ax.scatter(c_uniq, da_r_emp, s=sizes,
                                       color='#4e9af1', alpha=0.85, marker='x',
                                       label='recovered (sim)')
                            ax.plot(c_grid, curve_t, color='#000000', lw=1.5,
                                    label='truth (analytic)')
                            ax.plot(c_grid, curve_r, color='#4e9af1', lw=1.5,
                                    ls='--', label='recovered (analytic)')
                            ax.axvline(0, color=_tc['text'], lw=0.5,
                                       ls='--', alpha=0.4)
                            if wi == 0:
                                _style_legend(ax, fontsize=8, loc='best')

                        fig_d.suptitle(
                            'Dopamine quartiles: truth vs recovered',
                            color=_tc['text'], fontsize=12)
                        fig_d.tight_layout()
                        _show_fig(fig_d, 'recovery_dopamine_quartiles.png')
                    else:
                        plt.close(fig_d)
            except Exception as exc:
                st.warning(f'Could not compute behavioural curves: {exc}')
        else:
            st.info(_skip_reason)
