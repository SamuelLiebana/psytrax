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
_DDM_EXACT_PARAMS = {'w', 'b', 'a', 'z'}
_DDM_APPROX_PARAMS = {'w', 'b', 'z'}
_LOGISTIC_PARAMS = {'w', 'b'}
_RT_CURVE_FAMILIES = {'race', 'ddm_exact', 'ddm_approx'}


# ---------------------------------------------------------------------------
# Theme-aware colours
# ---------------------------------------------------------------------------

def _is_dark_theme():
    """Detect whether Streamlit is using a dark theme."""
    # 1. Explicit theme.base set in config.toml or via st.set_page_config
    try:
        base = st.get_option('theme.base')
        if base is not None:
            return base == 'dark'
    except Exception:
        pass
    # 2. Runtime theme info injected by the Streamlit frontend
    if hasattr(st, 'session_state') and '_stcore_theme' in st.session_state:
        theme_info = st.session_state['_stcore_theme']
        if 'base' in theme_info:
            return theme_info['base'] == 'dark'
        # Check backgroundColor luminance as a heuristic
        bg = theme_info.get('backgroundColor', '')
        if bg.startswith('#') and len(bg) == 7:
            r, g, b = int(bg[1:3], 16), int(bg[3:5], 16), int(bg[5:7], 16)
            return (0.299 * r + 0.587 * g + 0.114 * b) < 128
    # Default to dark (Streamlit's default)
    return True


def _theme_colours():
    """Return a dict of colours that adapt to the current Streamlit theme."""
    if _is_dark_theme():
        return {
            'bg':         '#0e1117',
            'text':       'white',
            'spine':      '#333333',
            'legend_bg':  '#1a1a2e',
            'legend_edge':'#333333',
            'grid':       'white',
        }
    else:
        return {
            'bg':         'white',
            'text':       'black',
            'spine':      '#cccccc',
            'legend_bg':  '#f0f0f0',
            'legend_edge':'#cccccc',
            'grid':       'black',
        }


def _style_fig(fig):
    """Set the figure background to match the Streamlit theme."""
    tc = _theme_colours()
    fig.patch.set_facecolor(tc['bg'])
    return tc


def _style_legend(ax, **kwargs):
    """Apply theme-aware styling to an axis legend."""
    tc = _theme_colours()
    ax.legend(
        facecolor=tc['legend_bg'],
        edgecolor=tc['legend_edge'],
        labelcolor=tc['text'],
        **kwargs,
    )


def _style_ax(ax, xlabel=None, ylabel=None, title=None):
    tc = _theme_colours()
    ax.set_facecolor(tc['bg'])
    ax.tick_params(colors=tc['text'])
    for spine in ax.spines.values():
        spine.set_edgecolor(tc['spine'])
    if xlabel:
        ax.set_xlabel(xlabel, color=tc['text'])
    if ylabel:
        ax.set_ylabel(ylabel, color=tc['text'])
    if title:
        ax.set_title(title, color=tc['text'])


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


def _mlp_psychometric(params_window, param_names, c_grid):
    """P(right|c) for the MLP model over a contrast grid.

    Varies contrast; any additional inputs are held at zero.
    """
    mp   = np.mean(params_window, axis=1)
    n_W1 = sum(1 for p in param_names if p.startswith('W1_'))
    H    = sum(1 for p in param_names if p.startswith('b1_'))
    n_in = n_W1 // H
    W1 = mp[:n_W1].reshape(n_in, H)
    b1 = mp[n_W1:n_W1 + H]
    W2 = mp[n_W1 + H:n_W1 + 2 * H]
    b2 = mp[-1]

    p_right = np.zeros(len(c_grid))
    for i, c in enumerate(c_grid):
        x      = np.zeros(n_in)
        x[0]   = c                          # first input is always contrast
        h      = np.tanh(W1.T @ x + b1)
        logit  = W2 @ h + b2
        p_right[i] = 1.0 / (1.0 + np.exp(-logit))
    return p_right


def _race_curves(params_window, param_names, c_grid, fixed_params=None, t_max=30.0, n_t=2000):
    """Compute P(right|c) and E[min(T_R,T_L)|c] for the race model.

    Uses the mean of params over the window for a deterministic prediction.
    Integrates numerically over a time grid using the trapezoidal rule.
    """
    mp = np.mean(params_window, axis=1)
    idx = {name: i for i, name in enumerate(param_names)}
    wr  = mp[idx['wr']];  wl  = mp[idx['wl']]
    br  = mp[idx['br']];  bl  = mp[idx['bl']]
    z   = mp[idx['z']]
    if 'sig_i' in idx:
        si = mp[idx['sig_i']]
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


def _ddm_exact_curves(params_window, param_names, c_grid):
    """Compute psychometric and chronometric predictions for the exact DDM."""
    mp = np.mean(params_window, axis=1)
    idx = {name: i for i, name in enumerate(param_names)}
    w = float(mp[idx['w']])
    b = float(mp[idx['b']])
    a = max(float(mp[idx['a']]), 1e-6)
    z_rel = float(np.clip(mp[idx['z']], 1e-6, 1.0 - 1e-6))
    z_abs = a * z_rel

    drift = w * np.asarray(c_grid, dtype=float) + b
    p_right = _ddm_exact_hit_prob(drift, a, z_abs)

    mean_rts = np.empty_like(drift, dtype=float)
    near_zero = np.isclose(drift, 0.0, atol=1e-8)
    mean_rts[near_zero] = z_abs * (a - z_abs)
    mean_rts[~near_zero] = (a * p_right[~near_zero] - z_abs) / drift[~near_zero]
    mean_rts = np.where(np.isfinite(mean_rts), np.maximum(mean_rts, 0.0), np.nan)
    return p_right, mean_rts


def _ddm_approx_curves(params_window, param_names, c_grid, n_t=2000):
    """Compute psychometric and chronometric predictions for the approx DDM."""
    mp = np.mean(params_window, axis=1)
    idx = {name: i for i, name in enumerate(param_names)}
    w = float(mp[idx['w']])
    b = float(mp[idx['b']])
    z = max(float(mp[idx['z']]), 1e-6)

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
    fixed_params = (result or {}).get('fixed_params') or {}
    param_set = set(param_names)
    if param_set == _RACE_FULL_PARAMS or (param_set == _RACE_DYNAMIC_PARAMS and 'sig_i' in fixed_params):
        return 'race', fixed_params
    if param_set == _DDM_EXACT_PARAMS:
        return 'ddm_exact', fixed_params
    if param_set == _DDM_APPROX_PARAMS:
        return 'ddm_approx', fixed_params
    if param_set == _LOGISTIC_PARAMS:
        return 'logistic', fixed_params
    if _is_mlp(param_names):
        return 'mlp', fixed_params
    return 'unknown', fixed_params


def _curve_predictions(params_window, param_names, c_grid, model_family, fixed_params=None):
    if model_family == 'race':
        return _race_curves(params_window, param_names, c_grid, fixed_params=fixed_params)
    if model_family == 'ddm_exact':
        return _ddm_exact_curves(params_window, param_names, c_grid)
    if model_family == 'ddm_approx':
        return _ddm_approx_curves(params_window, param_names, c_grid)
    if model_family == 'logistic':
        mp = np.mean(params_window, axis=1)
        iw = param_names.index('w')
        ib = param_names.index('b')
        psych = 1.0 / (1.0 + np.exp(-(mp[iw] * c_grid + mp[ib])))
        return psych, None
    if model_family == 'mlp':
        return _mlp_psychometric(params_window, param_names, c_grid), None
    return None, None


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

st.set_page_config(page_title='psytrax', layout='wide')

_APP_DIR = os.path.dirname(__file__)
_DOC_ASSET_DIR = os.path.join(_APP_DIR, 'examples', 'assets')
_DAP011_TRAJ = os.path.join(_DOC_ASSET_DIR, 'dap011_race_trajectories.png')
_DAP011_PSY = os.path.join(_DOC_ASSET_DIR, 'dap011_race_psychometric.png')
_DAP011_CHRONO = os.path.join(_DOC_ASSET_DIR, 'dap011_race_chronometric.png')

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title('psytrax')
st.sidebar.caption('Empirical Bayes for trial-by-trial decision models')
page = st.sidebar.radio('Navigation', ['Instructions', 'Fit Model', 'Visualise Results', 'Compare Models'],
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
    st.caption('Example output from the bundled DAP011 race-model fit.')
    if all(os.path.exists(path) for path in (_DAP011_TRAJ, _DAP011_PSY, _DAP011_CHRONO)):
        st.image(_DAP011_TRAJ, caption='Parameter trajectories recovered across learning',
                 use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.image(_DAP011_PSY, caption='Psychometric evolution over learning',
                     use_container_width=True)
        with col2:
            st.image(_DAP011_CHRONO, caption='Chronometric evolution over learning',
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
measured on bundled example mice spanning lower and higher trial counts.
Times include warm-start, hyperparameter optimisation, and Hessian computation.

| Model | **DAP044** (520) | **DAP027** (2,535) | **DAP009** (5,289) | **DAP011** (5,681) | **DAP031** (7,836) | **DAP007** (11,411) |
|---|---|---|---|---|---|---|
| Logistic (K=2) | 2.1s | 10.8s | 10.3s | 23.0s | 19.1s | 20.3s |
| DDM approx (K=3) | 5.3s | 11.1s | 28.4s | 45.6s | 40.5s | 35.8s |
| Race fixed sig_i (K=5) | 12.6s | 28.2s | 13.9s | 76.9s | 72.8s | 101.2s |

These timings use real datasets rather than synthetic sweeps, so they are not
perfectly monotonic in trial count. NVIDIA CUDA (float64) is expected to give
a further **3–8× speedup** for models with K ≥ 3 via `jax.vmap` on-device computation.
""")

    st.subheader('Installation')
    st.markdown("""
Core package:

```bash
pip install -e .
```

Streamlit app and plotting dependencies:

```bash
pip install -e .[web]
streamlit run app.py
```
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
        ['Example data (26 mice)', 'Upload my own file'],
        horizontal=True,
        key='fit_data_source',
    )

    import pandas as pd

    if data_source == 'Example data (26 mice)':
        _data_dir = os.path.join(os.path.dirname(__file__), 'data')
        _available = sorted(
            f.replace('_data.npy', '')
            for f in os.listdir(_data_dir)
            if f.endswith('_data.npy')
        ) if os.path.isdir(_data_dir) else []

        if not _available:
            st.error('No example data found in `data/`. Run `extract_data.py` first.')
            st.stop()

        animal = st.selectbox('Select animal', _available, key='fit_animal')
        raw = np.load(os.path.join(_data_dir, f'{animal}_data.npy'), allow_pickle=True).item()

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

    st.divider()

    # --- Model selection ---
    st.subheader('2. Choose model')
    model_choice = st.selectbox(
        'Built-in model',
        ['Race model (inverse-Gaussian)', 'DDM — exact (Navarro & Fuss 2009)',
         'DDM — approx (inverse-Gaussian)', 'Logistic regression'],
        key='fit_model',
    )

    if model_choice == 'Race model (inverse-Gaussian)':
        from psytrax.models.race import (
            log_lik_trial as _race_full_llt,
            make_fixed_sig_i_model as _make_fixed_sig_i_model,
            default_hyper_fixed_sig_i as _race_fixed_dhyper,
            DEFAULT_FIXED_SIG_I as _RACE_FIXED_SIG_I,
            DATA_SPEC as _data_spec,
        )
        _race_fixed_sig_i = True
        _llt = _race_full_llt
        _K = 5
        _pnames = ['wr', 'wl', 'br', 'bl', 'z']
        _dhyper = _race_fixed_dhyper

        st.markdown("""
**Race model** — two independent inverse-Gaussian accumulators racing to threshold.
`sig_i` is held fixed over learning at a built-in nuisance value.
The learning fit therefore uses 5 trial-varying parameters: `wr, wl, br, bl, z`.
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
4 parameters: `w` (contrast weight), `b` (bias), `a` (boundary separation),
`z` (relative starting point, 0–1).
""")
        _race_fixed_sig_i = False
    elif model_choice == 'DDM — approx (inverse-Gaussian)':
        from psytrax.models.ddm_approx import (
            log_lik_trial as _llt,
            N_PARAMS as _K,
            PARAM_NAMES as _pnames,
            default_hyper as _dhyper,
            DATA_SPEC as _data_spec,
        )
        st.markdown("""
**DDM (approx)** — single-accumulator inverse-Gaussian approximation (one absorbing
barrier). Faster than the exact DDM; accurate when error rates are low.
3 parameters: `w` (contrast weight), `b` (bias), `z` (threshold).
""")
        _race_fixed_sig_i = False
    else:
        import jax, jax.numpy as jnp

        def _llt(params, dat_trial):
            w, b = params
            p = jax.nn.sigmoid(w * dat_trial['inputs']['c'] + b)
            p = jnp.clip(p, 1e-7, 1 - 1e-7)
            return dat_trial['r'] * jnp.log(p) + (1 - dat_trial['r']) * jnp.log(1 - p)

        _K = 2
        _pnames = ['w', 'b']

        def _dhyper():
            return {'sigma': np.full(2, 2**-3), 'sigInit': np.full(2, 2**4), 'sigDay': None}

        from psytrax.models.logistic import DATA_SPEC as _data_spec
        st.markdown("""
**Logistic regression** — 2 parameters per trial: `w` (weight) and `b` (bias).
""")
        _race_fixed_sig_i = False

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
        # Build the learning rule callable (race model with fixed sig_i is
        # deferred until _run_fit where the wrapped likelihood is available).
        if model_choice == 'Race model (inverse-Gaussian)' and _race_fixed_sig_i:
            _learning_rule = 'reinforce_deferred'
        else:
            _learning_rule = make_reinforce(_llt, reward_key=_lr_reward_col)

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
        hess_calc = st.selectbox('Credible intervals', ['weights', 'None', 'hyper', 'All'],
                                 index=0, key='fit_hess')
        hess_calc = None if hess_calc == 'None' else hess_calc
        precision = 'float64'
        fixed_sig_i = _RACE_FIXED_SIG_I if model_choice == 'Race model (inverse-Gaussian)' and _race_fixed_sig_i else None

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

                fixed_params = {}
                _lr_actual = _learning_rule  # may be callable, 'reinforce_deferred', or None

                if model_choice == 'Race model (inverse-Gaussian)' and _race_fixed_sig_i:
                    fixed_params['sig_i'] = float(fixed_sig_i)
                    _status_cb({'stage': 'setup', 'message': f'Using fixed sig_i = {fixed_sig_i:.4f}.'})
                    _llt_fit, _K_fit, _pnames_fit, _dhyper_fit, _ = _make_fixed_sig_i_model(fixed_sig_i)
                    fit_kwargs.update(
                        log_lik_trial=_llt_fit,
                        n_params=_K_fit,
                        param_names=_pnames_fit,
                    )
                    # Build REINFORCE rule for the fixed-sig_i wrapper
                    if _lr_actual == 'reinforce_deferred':
                        from psytrax.learning_rules import make_reinforce
                        _lr_actual = make_reinforce(_llt_fit, reward_key=_lr_reward_col)
                else:
                    fit_kwargs.update(
                        log_lik_trial=_llt,
                        n_params=_K,
                        param_names=_pnames,
                    )

                # Pass learning rule if one was selected
                if _lr_actual and _lr_actual != 'reinforce_deferred':
                    fit_kwargs['learning_rule'] = _lr_actual
                    _status_cb({'stage': 'setup', 'message': 'Learning rule enabled — alpha will be optimised.'})

                result = psytrax.fit(
                    hyper=hyper,
                    **fit_kwargs,
                )
                if fixed_params:
                    _saved = np.load(result, allow_pickle=True).item()
                    _saved['fixed_params'] = fixed_params
                    np.save(result, _saved)
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

        # Poll the queue while the thread is alive, streaming updates to the browser
        cycle, map_iter, log_evd_str, best_str, map_loss_str = 0, 0, '—', '—', '—'
        current_status = 'Preparing fit…'
        fit_log = st.session_state.get('fit_log', [])
        while _thread.is_alive():
            while not _q.empty():
                try:
                    msg = _q.get_nowait()
                    if msg[0] == 'progress':
                        _, cycle, map_iter, postfix = msg
                        log_evd_str  = postfix.get('log_evd',  '—')
                        best_str     = postfix.get('best',     '—')
                        map_loss_str = postfix.get('MAP loss', map_loss_str)
                    elif msg[0] == 'status':
                        payload = msg[1]
                        current_status = payload.get('message', current_status)
                        fit_log.append(current_status)
                        fit_log = fit_log[-12:]
                        st.session_state['fit_log'] = fit_log
                except queue.Empty:
                    break
            cycle_text.metric('Cycles completed', cycle)
            map_text.metric('MAP iters (current cycle)', map_iter)
            status_text.markdown(f'**Current step:** {current_status}')
            log_evd_text.markdown(
                f'Log evidence (higher is better) — current: **{log_evd_str}** &nbsp;|&nbsp; best: **{best_str}**'
                + (
                    f' &nbsp;|&nbsp; Neg. log posterior (lower is better): **{map_loss_str}**'
                    if map_loss_str != '—' else ''
                )
            )
            if fit_log:
                log_box.code('\n'.join(fit_log), language='text')
            time.sleep(0.5)

        # Thread finished — drain any remaining messages
        msg_type, payload = 'error', 'No result received from fitting thread.'
        while not _q.empty():
            try:
                m = _q.get_nowait()
                if m[0] in ('done', 'error'):
                    msg_type, payload = m[0], m[1]
            except queue.Empty:
                break

        st.session_state['fit_running'] = False
        if msg_type == 'done':
            st.session_state['fit_result_path'] = payload
        else:
            st.session_state['fit_error'] = payload
        st.rerun()

    if st.session_state['fit_error']:
        st.error(f'Fitting failed:\n\n```\n{st.session_state["fit_error"]}\n```')

    if st.session_state['fit_result_path']:
        path = st.session_state['fit_result_path']
        st.success(f'Fit complete! Saved to `{path}`')
        res = np.load(path, allow_pickle=True).item()
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

        with open(path, 'rb') as f:
            st.download_button(
                'Download fit file (.npy)',
                data=f.read(),
                file_name=os.path.basename(path),
                mime='application/octet-stream',
                key='fit_download',
            )
        st.info('Load this file in **Visualise Results** or **Compare Models** to explore the fit.')

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
    traj_mode = st.radio('Display mode', ['Separate', 'Combined'],
                         horizontal=True, label_visibility='collapsed')

    trials      = np.arange(N)
    day_lengths = dat.get('dayLength') if dat.get('dayLength') is not None else np.array([])
    boundaries  = np.cumsum(day_lengths).astype(int) if len(day_lengths) else np.array([], dtype=int)

    if traj_mode == 'Separate':
        n_cols = min(K, 3)
        n_rows = int(np.ceil(K / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False)
        _tc = _style_fig(fig)
        for k, (ax, name) in enumerate(zip(axes.flat, param_names)):
            col = COLORS[k % len(COLORS)]
            _style_ax(ax, xlabel='Trial', title=name)
            ax.plot(trials, params[k], color=col, lw=0.8, alpha=0.9)
            if W_std is not None:
                ax.fill_between(trials, params[k] - W_std[k], params[k] + W_std[k],
                                color=col, alpha=0.2)
            for b in boundaries[:-1]:
                ax.axvline(b, color=_tc['text'], lw=0.5, alpha=0.3, ls='--')
        for ax in axes.flat[K:]:
            ax.set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:  # Combined
        fig, ax = plt.subplots(figsize=(12, 4))
        _tc = _style_fig(fig)
        _style_ax(ax, xlabel='Trial', ylabel='Parameter value')
        for k, name in enumerate(param_names):
            col = COLORS[k % len(COLORS)]
            ax.plot(trials, params[k], color=col, lw=0.9, alpha=0.9, label=name)
            if W_std is not None:
                ax.fill_between(trials, params[k] - W_std[k], params[k] + W_std[k],
                                color=col, alpha=0.15)
        for b in boundaries[:-1]:
            ax.axvline(b, color=_tc['text'], lw=0.5, alpha=0.3, ls='--')
        _style_legend(ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

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
        st.pyplot(fig_evo)
        plt.close(fig_evo)

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
                st.pyplot(fig_cevo)
                plt.close(fig_cevo)

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
    st.pyplot(fig)
    plt.close(fig)

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
            st.pyplot(fig_p)
            plt.close(fig_p)

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
                st.pyplot(fig_c)
                plt.close(fig_c)

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
                fig3.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)
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
                fig3.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)
