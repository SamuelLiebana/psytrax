"""psytrax web app — instructions, fit model, result visualiser.

Run with:  streamlit run app.py
"""

import io
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

def _style_ax(ax, xlabel=None, ylabel=None, title=None):
    ax.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    if xlabel:
        ax.set_xlabel(xlabel, color='white')
    if ylabel:
        ax.set_ylabel(ylabel, color='white')
    if title:
        ax.set_title(title, color='white')


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


def _race_curves(params_window, param_names, c_grid, t_max=30.0, n_t=2000):
    """Compute P(right|c) and E[min(T_R,T_L)|c] for the race model.

    Uses the mean of params over the window for a deterministic prediction.
    Integrates numerically over a time grid using the trapezoidal rule.
    """
    mp = np.mean(params_window, axis=1)
    idx = {name: i for i, name in enumerate(param_names)}
    wr  = mp[idx['wr']];  wl  = mp[idx['wl']]
    br  = mp[idx['br']];  bl  = mp[idx['bl']]
    z   = mp[idx['z']];   si  = mp[idx['sig_i']]

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
        p_rights[i] = np.clip(np.trapz(f1 * (1 - F2), t_grid), 0.0, 1.0)
        # E[min(T_R,T_L)] = ∫ (1−F_R)(1−F_L) dt
        mean_rts[i] = max(np.trapz((1 - F1) * (1 - F2), t_grid), 0.0)

    return p_rights, mean_rts

st.set_page_config(page_title='psytrax', layout='wide')

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
JAX automatically uses a GPU if one is available. Install the right backend first:

| Platform | Command |
|----------|---------|
| Apple Silicon (Metal) | `pip install jax-metal` |
| NVIDIA CUDA 12 | `pip install jax[cuda12]` |
| NVIDIA CUDA 11 | `pip install jax[cuda11_pip]` |

Then pass `device='gpu'` (or `'auto'`, the default) to `psytrax.fit()`.
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
| `responses` | `r` | `array (N,)` | Integer responses (e.g. 0/1) |
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
| `responses` | **Yes** | Binary response column (0 / 1 or Left / Right) |
| `times` | No | Reaction-time column (seconds) |
| `session_id` | No | Column whose value identifies the session — used to compute session lengths |
""")

        data_file = st.file_uploader('Data file (.npy or .csv)', type=['npy', 'csv'], key='fit_data')
        if data_file is None:
            st.info('Upload a `.npy` or `.csv` file to continue.')
            st.stop()

    if data_source == 'Upload my own file' and data_file.name.endswith('.csv'):
        df = pd.read_csv(data_file)
        st.dataframe(df.head(5), use_container_width=True)
        cols = ['— none —'] + list(df.columns)
        num_cols = ['— none —'] + [c for c in df.columns
                                    if pd.api.types.is_numeric_dtype(df[c])]

        st.markdown('**Map columns to psytrax fields:**')
        mc1, mc2 = st.columns(2)
        with mc1:
            input_cols = st.multiselect(
                'Input columns (model inputs)', df.columns.tolist(),
                default=[c for c in df.columns
                         if c.lower() in ('c', 'contrast', 'signed_contrast')],
                key='csv_inputs',
            )
            resp_col = st.selectbox('Response column', cols, key='csv_resp',
                                    index=next((i for i, c in enumerate(cols)
                                                if c.lower() in ('r', 'response', 'responses',
                                                                  'choice')), 0))
        with mc2:
            rt_col = st.selectbox('Reaction-time column (optional)', num_cols,
                                  key='csv_rt',
                                  index=next((i for i, c in enumerate(num_cols)
                                              if 'time' in c.lower() or c.lower() == 't'), 0))
            sess_col = st.selectbox('Session-ID column (optional)', cols,
                                    key='csv_sess',
                                    index=next((i for i, c in enumerate(cols)
                                                if 'sess' in c.lower() or 'day' in c.lower()), 0))

        if not input_cols:
            st.warning('Select at least one input column.')
            st.stop()
        if resp_col == '— none —':
            st.warning('Select a response column.')
            st.stop()

        # Build responses (handle text labels)
        resp_raw = df[resp_col]
        if pd.api.types.is_numeric_dtype(resp_raw):
            resp_arr = resp_raw.to_numpy(dtype=float)
        else:
            unique_vals = resp_raw.dropna().unique()
            if len(unique_vals) != 2:
                st.error(f'Response column has {len(unique_vals)} unique values; expected 2.')
                st.stop()
            # Map alphabetically: lower value → 0, higher → 1
            unique_sorted = sorted(unique_vals)
            st.info(f'Mapping responses: `{unique_sorted[0]}` → 0, `{unique_sorted[1]}` → 1')
            resp_arr = resp_raw.map({unique_sorted[0]: 0.0, unique_sorted[1]: 1.0}).to_numpy(dtype=float)

        # Drop rows with NaN in required columns
        keep_mask = np.isfinite(resp_arr)
        for ic in input_cols:
            if pd.api.types.is_numeric_dtype(df[ic]):
                keep_mask &= np.isfinite(df[ic].to_numpy(dtype=float))
        if rt_col != '— none —':
            keep_mask &= np.isfinite(df[rt_col].to_numpy(dtype=float))
        if keep_mask.sum() < len(df):
            st.warning(f'Dropped {len(df) - keep_mask.sum()} rows with NaN/non-finite values.')
        df_clean = df[keep_mask].reset_index(drop=True)
        resp_arr = resp_arr[keep_mask]

        raw = {
            'inputs': {ic: df_clean[ic].to_numpy(dtype=float) for ic in input_cols},
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

    elif data_source == 'Upload my own file':
        raw = np.load(data_file, allow_pickle=True).item()  # .npy upload

    # Summary preview
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
            log_lik_trial as _llt,
            N_PARAMS as _K,
            PARAM_NAMES as _pnames,
            default_hyper as _dhyper,
        )
        st.markdown("""
**Race model** — two independent inverse-Gaussian accumulators racing to threshold.
6 parameters: `wr, wl, br, bl, z, sig_i`.
Expects `inputs['c']` and `times`. Subtract non-decision time from RTs before uploading.
""")
    elif model_choice == 'DDM — exact (Navarro & Fuss 2009)':
        from psytrax.models.ddm import (
            log_lik_trial as _llt,
            N_PARAMS as _K,
            PARAM_NAMES as _pnames,
            default_hyper as _dhyper,
        )
        st.markdown("""
**DDM (exact)** — Wiener process between two absorbing barriers, using the
Navarro & Fuss (2009) / Bogacz et al. (2006) hybrid series solution.
4 parameters: `w` (contrast weight), `b` (bias), `a` (boundary separation),
`z` (relative starting point, 0–1).
Expects `inputs['c']` and `times`. Subtract non-decision time from RTs before uploading.
""")
    elif model_choice == 'DDM — approx (inverse-Gaussian)':
        from psytrax.models.ddm_approx import (
            log_lik_trial as _llt,
            N_PARAMS as _K,
            PARAM_NAMES as _pnames,
            default_hyper as _dhyper,
        )
        st.markdown("""
**DDM (approx)** — single-accumulator inverse-Gaussian approximation (one absorbing
barrier). Faster than the exact DDM; accurate when error rates are low.
3 parameters: `w` (contrast weight), `b` (bias), `z` (threshold).
Expects `inputs['c']` and `times`. Subtract non-decision time from RTs before uploading.
""")
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

        st.markdown("""
**Logistic regression** — 2 parameters per trial: `w` (weight) and `b` (bias).

Expects `inputs['c']` (signed contrast) in your data.
""")

    st.divider()

    # --- Fitting options ---
    st.subheader('3. Configure fitting')

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
    hyper['sigma'] = custom_sigma

    st.divider()

    # --- Run ---
    st.subheader('4. Fit')

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

        _orig_tqdm = _hyper_opt_mod.tqdm
        _hyper_opt_mod.tqdm = _QueueTqdm

        def _run_fit():
            try:
                os.makedirs('fits', exist_ok=True)
                result = psytrax.fit(
                    data=raw,
                    log_lik_trial=_llt,
                    n_params=_K,
                    param_names=_pnames,
                    hyper=hyper,
                    shared_sigma=shared_sigma,
                    session_boundaries=session_boundaries,
                    n_trials=n_trials_opt,
                    hess_calc=hess_calc,
                    map_tol=float(map_tol),
                    precision=precision,
                    subject_name=subject_name,
                    save=True,
                    verbose=True,
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

        st.markdown('**Fitting in progress…** &nbsp; `JAX L-BFGS · float64`')
        col_cyc, col_map = st.columns(2)
        cycle_text   = col_cyc.empty()
        map_text     = col_map.empty()
        log_evd_text = st.empty()

        # Poll the queue while the thread is alive, streaming updates to the browser
        cycle, map_iter, log_evd_str, best_str, map_loss_str = 0, 0, '—', '—', '—'
        while _thread.is_alive():
            while not _q.empty():
                try:
                    msg = _q.get_nowait()
                    if msg[0] == 'progress':
                        _, cycle, map_iter, postfix = msg
                        log_evd_str  = postfix.get('log_evd',  '—')
                        best_str     = postfix.get('best',     '—')
                        map_loss_str = postfix.get('MAP loss', map_loss_str)
                except queue.Empty:
                    break
            cycle_text.metric('Cycles completed', cycle)
            map_text.metric('MAP iters (current cycle)', map_iter)
            log_evd_text.markdown(
                f'Log evidence — current: **{log_evd_str}** &nbsp;|&nbsp; best: **{best_str}**'
                + (f' &nbsp;|&nbsp; MAP loss: **{map_loss_str}**' if map_loss_str != '—' else '')
            )
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
    RACE_PARAMS     = {'wr', 'wl', 'br', 'bl', 'z', 'sig_i'}
    LOGISTIC_PARAMS = {'w', 'b'}
    param_set  = set(param_names)
    is_race    = param_set == RACE_PARAMS
    is_logistic= param_set == LOGISTIC_PARAMS
    is_mlp     = _is_mlp(param_names)
    # Locate RT array (stored as 'T' or 'times')
    _rt_key = next((k for k in ('T', 'times') if k in dat and dat[k] is not None), None)
    has_rt  = (_rt_key is not None) and is_race

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
        fig.patch.set_facecolor('#0e1117')
        for k, (ax, name) in enumerate(zip(axes.flat, param_names)):
            col = COLORS[k % len(COLORS)]
            _style_ax(ax, xlabel='Trial', title=name)
            ax.plot(trials, params[k], color=col, lw=0.8, alpha=0.9)
            if W_std is not None:
                ax.fill_between(trials, params[k] - W_std[k], params[k] + W_std[k],
                                color=col, alpha=0.2)
            for b in boundaries[:-1]:
                ax.axvline(b, color='white', lw=0.5, alpha=0.3, ls='--')
        for ax in axes.flat[K:]:
            ax.set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:  # Combined
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor('#0e1117')
        _style_ax(ax, xlabel='Trial', ylabel='Parameter value')
        for k, name in enumerate(param_names):
            col = COLORS[k % len(COLORS)]
            ax.plot(trials, params[k], color=col, lw=0.9, alpha=0.9, label=name)
            if W_std is not None:
                ax.fill_between(trials, params[k] - W_std[k], params[k] + W_std[k],
                                color=col, alpha=0.15)
        for b in boundaries[:-1]:
            ax.axvline(b, color='white', lw=0.5, alpha=0.3, ls='--')
        ax.legend(facecolor='#1a1a2e', edgecolor='#333333', labelcolor='white')
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
        fig_evo.patch.set_facecolor('#0e1117')

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
                       color='white', zorder=3)

            params_win = params[:, t0:t1]
            if is_race:
                p_m, _ = _race_curves(params_win, param_names, c_grid)
                ax.plot(c_grid, p_m, color='#4e9af1', lw=2)
            elif is_logistic:
                iw = param_names.index('w'); ib = param_names.index('b')
                w_m = np.mean(params_win[iw]); b_m = np.mean(params_win[ib])
                ax.plot(c_grid, 1 / (1 + np.exp(-(w_m * c_grid + b_m))),
                        color='#4e9af1', lw=2)
            elif is_mlp:
                ax.plot(c_grid, _mlp_psychometric(params_win, param_names, c_grid),
                        color='#4e9af1', lw=2)

            ax.axhline(0.5, color='white', lw=0.5, ls='--', alpha=0.4)
            ax.axvline(0,   color='white', lw=0.5, ls='--', alpha=0.4)
            ax.set_ylim(0, 1)

        fig_evo.suptitle('Psychometric curve evolution', color='white', fontsize=13)
        fig_evo.tight_layout()
        st.pyplot(fig_evo)
        plt.close(fig_evo)

        # --- Chronometric evolution (race model + RT data only) ---
        if has_rt:
            T_data = dat[_rt_key]
            st.subheader('Chronometric curve: evolution over learning')
            with st.spinner('Computing chronometric curves…'):
                fig_cevo, axes_cevo = plt.subplots(2, 2, figsize=(11, 8))
                fig_cevo.patch.set_facecolor('#0e1117')

                for wi, ax in enumerate(axes_cevo.flat):
                    t0, t1 = int(edges[wi]), int(edges[wi + 1])
                    mask   = np.zeros(N, dtype=bool)
                    mask[t0:t1] = True

                    c_win  = c_data[mask]
                    rt_win = T_data[mask]
                    c_uniq_win  = np.unique(c_win)
                    rt_win_mean = np.array([rt_win[c_win == cv].mean() for cv in c_uniq_win])
                    n_win       = np.array([np.sum(c_win == cv) for cv in c_uniq_win])

                    _style_ax(ax, xlabel='Signed contrast', ylabel='Mean RT (s)',
                              title=f'Trials {t0 + 1}–{t1}')
                    ax.scatter(c_uniq_win, rt_win_mean, s=[max(10, n / 5) for n in n_win],
                               color='white', zorder=3)

                    _, rt_m = _race_curves(params[:, t0:t1], param_names, c_grid)
                    ax.plot(c_grid, rt_m, color='#4e9af1', lw=2)
                    ax.axvline(0, color='white', lw=0.5, ls='--', alpha=0.4)

                fig_cevo.suptitle('Chronometric curve evolution', color='white', fontsize=13)
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
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    colors = ['#4e9af1', '#f1a44e', '#4ef17a', '#f14e7a', '#c44ef1']
    bars = ax.bar(names, evds, color=[colors[i % len(colors)] for i in range(len(names))])
    ax.bar_label(bars, fmt='%.1f', color='white', padding=4)
    ax.set_ylabel('Log evidence', color='white')
    ax.tick_params(colors='white', axis='both')
    ax.set_xticklabels(names, rotation=20, ha='right')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
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
        any_race   = any(set(r['param_names']) == {'wr', 'wl', 'br', 'bl', 'z', 'sig_i'}
                         for r in results.values())

        # --- Psychometric evolution ---
        st.subheader('Psychometric curve: evolution over learning')
        with st.spinner('Computing psychometric curves…'):
            fig_p, axes_p = plt.subplots(2, 2, figsize=(11, 8))
            fig_p.patch.set_facecolor('#0e1117')

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
                           color='white', zorder=3, label='data')

                for mi, (mname, res) in enumerate(results.items()):
                    pn  = res['param_names']
                    par = res['params'][:, t0:t1]
                    col = colors[mi % len(colors)]
                    if set(pn) == {'wr', 'wl', 'br', 'bl', 'z', 'sig_i'}:
                        p_m, _ = _race_curves(par, pn, c_grid)
                        ax.plot(c_grid, p_m, color=col, lw=2, label=mname)
                    elif set(pn) == {'w', 'b'}:
                        iw = pn.index('w'); ib = pn.index('b')
                        w_m = np.mean(par[iw]); b_m = np.mean(par[ib])
                        ax.plot(c_grid, 1 / (1 + np.exp(-(w_m * c_grid + b_m))),
                                color=col, lw=2, label=mname)
                    elif _is_mlp(pn):
                        ax.plot(c_grid, _mlp_psychometric(par, pn, c_grid),
                                color=col, lw=2, label=mname)

                ax.axhline(0.5, color='white', lw=0.5, ls='--', alpha=0.4)
                ax.axvline(0,   color='white', lw=0.5, ls='--', alpha=0.4)
                ax.set_ylim(0, 1)
                ax.legend(facecolor='#1a1a2e', edgecolor='#333333', labelcolor='white',
                          fontsize=7)

            fig_p.suptitle('Psychometric curve evolution', color='white', fontsize=13)
            fig_p.tight_layout()
            st.pyplot(fig_p)
            plt.close(fig_p)

        # --- Chronometric evolution (race models + RT data) ---
        if _rt_key_cm is not None and any_race:
            T_data = dat[_rt_key_cm]
            st.subheader('Chronometric curve: evolution over learning')
            with st.spinner('Computing chronometric curves…'):
                fig_c, axes_c = plt.subplots(2, 2, figsize=(11, 8))
                fig_c.patch.set_facecolor('#0e1117')

                for wi, ax in enumerate(axes_c.flat):
                    t0, t1 = int(edges[wi]), int(edges[wi + 1])
                    mask   = np.zeros(N_cm, dtype=bool)
                    mask[t0:t1] = True

                    c_win  = c_data[mask]; rt_win = T_data[mask]
                    c_uniq_win  = np.unique(c_win)
                    rt_win_mean = np.array([rt_win[c_win == cv].mean() for cv in c_uniq_win])
                    n_win       = np.array([np.sum(c_win == cv) for cv in c_uniq_win])

                    _style_ax(ax, xlabel='Signed contrast', ylabel='Mean RT (s)',
                              title=f'Trials {t0 + 1}–{t1}')
                    ax.scatter(c_uniq_win, rt_win_mean, s=[max(10, n / 5) for n in n_win],
                               color='white', zorder=3, label='data')

                    for mi, (mname, res) in enumerate(results.items()):
                        pn  = res['param_names']
                        par = res['params'][:, t0:t1]
                        if set(pn) == {'wr', 'wl', 'br', 'bl', 'z', 'sig_i'}:
                            _, rt_m = _race_curves(par, pn, c_grid)
                            ax.plot(c_grid, rt_m, color=colors[mi % len(colors)],
                                    lw=2, label=mname)

                    ax.axvline(0, color='white', lw=0.5, ls='--', alpha=0.4)
                    ax.legend(facecolor='#1a1a2e', edgecolor='#333333', labelcolor='white',
                              fontsize=7)

                fig_c.suptitle('Chronometric curve evolution', color='white', fontsize=13)
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
                fig3.patch.set_facecolor('#0e1117')
                for k, (ax, pname) in enumerate(zip(axes.flat, param_names)):
                    _style_ax(ax, xlabel='Trial', title=pname)
                    ax.plot(trials, params[k], color=color, lw=0.8, alpha=0.9)
                    if W_std is not None:
                        ax.fill_between(trials, params[k] - W_std[k], params[k] + W_std[k],
                                        color=color, alpha=0.2)
                    for b in boundaries[:-1]:
                        ax.axvline(b, color='white', lw=0.5, alpha=0.3, ls='--')
                for ax in axes.flat[K:]:
                    ax.set_visible(False)
                fig3.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)
            else:  # Combined
                fig3, ax3 = plt.subplots(figsize=(12, 4))
                fig3.patch.set_facecolor('#0e1117')
                _style_ax(ax3, xlabel='Trial', ylabel='Parameter value')
                for k, pname in enumerate(param_names):
                    col_k = colors[k % len(colors)]
                    ax3.plot(trials, params[k], color=col_k, lw=0.9, alpha=0.9, label=pname)
                    if W_std is not None:
                        ax3.fill_between(trials, params[k] - W_std[k], params[k] + W_std[k],
                                         color=col_k, alpha=0.15)
                for b in boundaries[:-1]:
                    ax3.axvline(b, color='white', lw=0.5, alpha=0.3, ls='--')
                ax3.legend(facecolor='#1a1a2e', edgecolor='#333333', labelcolor='white',
                           fontsize=7)
                fig3.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)
