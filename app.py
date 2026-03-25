"""psytrax web app — instructions + result visualiser.

Run with:  streamlit run app.py
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

st.set_page_config(page_title='psytrax', layout='wide')

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title('psytrax')
st.sidebar.caption('Empirical Bayes for trial-by-trial decision models')
page = st.sidebar.radio('', ['Instructions', 'Visualise Results', 'Compare Models'])

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
# Visualise Results
# ---------------------------------------------------------------------------
else:
    st.title('Visualise Results')

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

    # --- Parameter trajectories ---
    st.subheader('Parameter trajectories')

    trials = np.arange(N)

    # Session boundary lines
    day_lengths = dat.get('dayLength') if dat.get('dayLength') is not None else np.array([])
    boundaries = np.cumsum(day_lengths).astype(int) if len(day_lengths) else np.array([], dtype=int)

    n_cols = min(K, 3)
    n_rows = int(np.ceil(K / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False)
    fig.patch.set_facecolor('#0e1117')

    for k, (ax, name) in enumerate(zip(axes.flat, param_names)):
        ax.set_facecolor('#0e1117')
        ax.plot(trials, params[k], color='#4e9af1', lw=0.8, alpha=0.9)
        if W_std is not None:
            ax.fill_between(trials,
                            params[k] - W_std[k],
                            params[k] + W_std[k],
                            color='#4e9af1', alpha=0.2)
        for b in boundaries[:-1]:
            ax.axvline(b, color='white', lw=0.5, alpha=0.3, ls='--')
        ax.set_title(name, color='white')
        ax.set_xlabel('Trial', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

    # Hide unused axes
    for ax in axes.flat[K:]:
        ax.set_visible(False)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Psychometric curve ---
    if 'inputs' in dat and 'c' in dat['inputs'] and 'r' in dat:
        st.subheader('Psychometric curve')
        c = dat['inputs']['c']
        r = dat['r']
        contrasts = np.unique(c)

        p_right = [r[c == cv].mean() for cv in contrasts]
        n_trials_per = [np.sum(c == cv) for cv in contrasts]

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor('#0e1117')
        ax2.set_facecolor('#0e1117')
        sc = ax2.scatter(contrasts, p_right, s=[max(10, n / 5) for n in n_trials_per],
                         color='#4e9af1', zorder=3)
        ax2.axhline(0.5, color='white', lw=0.5, ls='--', alpha=0.4)
        ax2.axvline(0, color='white', lw=0.5, ls='--', alpha=0.4)
        ax2.set_xlabel('Signed contrast', color='white')
        ax2.set_ylabel('P(right)', color='white')
        ax2.set_title('Psychometric curve (all trials)', color='white')
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#333333')
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # --- Hyperparameter table ---
    st.subheader('Optimised hyperparameters (log₂ scale)')
    import pandas as pd
    hyper_rows = []
    for key, val in hyper.items():
        if val is not None:
            arr = np.atleast_1d(val)
            hyper_rows.append({
                'hyperparameter': key,
                **{name: f'{np.log2(arr[i]):.3f}' for i, name in enumerate(param_names) if i < len(arr)}
            })
    if hyper_rows:
        st.dataframe(pd.DataFrame(hyper_rows).set_index('hyperparameter'), use_container_width=True)

# ---------------------------------------------------------------------------
# Compare Models
# ---------------------------------------------------------------------------
elif page == 'Compare Models':
    st.title('Compare Models')
    import pandas as pd

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

    # --- Shared psychometric curve overlay ---
    first_res = next(iter(results.values()))
    dat = first_res['data']
    if 'inputs' in dat and 'c' in dat['inputs'] and 'r' in dat:
        st.subheader('Psychometric curve (data)')
        c, r = dat['inputs']['c'], dat['r']
        contrasts   = np.unique(c)
        p_right     = [r[c == cv].mean() for cv in contrasts]
        n_per       = [np.sum(c == cv) for cv in contrasts]

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor('#0e1117')
        ax2.set_facecolor('#0e1117')
        ax2.scatter(contrasts, p_right,
                    s=[max(10, n / 5) for n in n_per],
                    color='white', zorder=3, label='data')
        ax2.axhline(0.5, color='white', lw=0.5, ls='--', alpha=0.3)
        ax2.axvline(0,   color='white', lw=0.5, ls='--', alpha=0.3)
        ax2.set_xlabel('Signed contrast', color='white')
        ax2.set_ylabel('P(right)',        color='white')
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#333333')
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # --- Parameter trajectories per model ---
    st.subheader('Parameter trajectories')
    for i, (name, res) in enumerate(results.items()):
        with st.expander(name, expanded=(i == 0)):
            params      = res['params']
            param_names = res['param_names']
            K, N        = params.shape
            W_std       = res['hess_info'].get('W_std')
            trials      = np.arange(N)

            day_lengths = res['data'].get('dayLength') or np.array([])
            boundaries  = np.cumsum(day_lengths).astype(int) if len(day_lengths) else np.array([], dtype=int)

            n_cols = min(K, 3)
            n_rows = int(np.ceil(K / n_cols))
            fig3, axes = plt.subplots(n_rows, n_cols,
                                      figsize=(5 * n_cols, 3 * n_rows),
                                      squeeze=False)
            fig3.patch.set_facecolor('#0e1117')
            color = colors[i % len(colors)]

            for k, (ax, pname) in enumerate(zip(axes.flat, param_names)):
                ax.set_facecolor('#0e1117')
                ax.plot(trials, params[k], color=color, lw=0.8, alpha=0.9)
                if W_std is not None:
                    ax.fill_between(trials,
                                    params[k] - W_std[k],
                                    params[k] + W_std[k],
                                    color=color, alpha=0.2)
                for b in boundaries[:-1]:
                    ax.axvline(b, color='white', lw=0.5, alpha=0.3, ls='--')
                ax.set_title(pname, color='white')
                ax.set_xlabel('Trial', color='white')
                ax.tick_params(colors='white')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#333333')

            for ax in axes.flat[K:]:
                ax.set_visible(False)

            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
