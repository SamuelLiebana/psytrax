# psytrax

**Empirical Bayes fitting for trial-by-trial decision models**

psytrax fits a Gaussian random-walk prior over a sequence of K parameters across N trials and optimises the prior variance (hyperparameters) by maximising the marginal likelihood (evidence) using the Laplace approximation.

It is model-agnostic: you supply a **per-trial log-likelihood function** and psytrax handles all the inference machinery.

---

## Installation

```bash
pip install -e .
```

Dependencies: `jax`, `numpy`, `scipy`, `tqdm`. For the web app: `streamlit`.

---

## Quick start

```python
import psytrax
from psytrax.models.race import log_lik_trial, N_PARAMS, PARAM_NAMES, default_hyper, default_E0

data = {
    'inputs':          {'c': contrast_array},  # signed contrast, shape (N,)
    'responses':        response_array,        # 0 / 1, shape (N,)
    'times':            rt_array,              # reaction times, shape (N,)
    'session_lengths':  day_length_array,      # trials per session, shape (n_sessions,)
}

result = psytrax.fit(
    data               = data,
    log_lik_trial      = log_lik_trial,
    n_params           = N_PARAMS,
    param_names        = PARAM_NAMES,
    hyper              = default_hyper(),
    E0                 = default_E0(N),
    session_boundaries = True,
)

print(result['params'].shape)   # (K, N)
print(result['log_evidence'])   # scalar
```

---

## Writing your own model

Provide any JAX-compatible per-trial function:

```python
import jax
import jax.numpy as jnp

def my_log_lik_trial(params, dat_trial):
    """
    params    : jnp array (K,)  — parameters for this trial
    dat_trial : dict — same keys as your data dict but scalar-valued per trial
                (psytrax vmaps over trials automatically)
    """
    w, b = params
    p = jax.nn.sigmoid(w * dat_trial['inputs']['x'] + b)
    return dat_trial['r'] * jnp.log(p) + (1 - dat_trial['r']) * jnp.log(1 - p)

result = psytrax.fit(data=data, log_lik_trial=my_log_lik_trial, n_params=2)
```

The function must be written with **`jax.numpy`** (not `numpy`) so that psytrax can differentiate through it to obtain the gradient and Hessian needed for MAP estimation and the Laplace approximation.

---

## Built-in models

| Model | File | K | RT? | Description |
|-------|------|---|-----|-------------|
| Logistic | `models/logistic.py` | 2 | No | Binary logistic regression |
| DDM | `models/ddm.py` | 3 | Yes | Drift diffusion model (fixed noise) |
| Race | `models/race.py` | 6 | Yes | Race model with separate accumulators |
| MLP | `models/mlp.py` | 13 | No | 1→4→1 MLP with tanh hidden layer |

Each model exposes: `log_lik_trial`, `N_PARAMS`, `PARAM_NAMES`, `default_hyper()`, `default_E0(N)`.

See `examples/compare_models_DAP009.py` for a full comparison on real mouse data.

---

## Data format

| Key | Alias | Type | Description |
|-----|-------|------|-------------|
| `inputs` | — | `dict` | Dict of input arrays, each `(N, ...)` |
| `responses` | `r` | `array (N,)` | Integer responses (e.g. 0/1) |
| `times` | `T` | `array (N,)` | Reaction times *(optional)* |
| `session_lengths` | `dayLength` | `array` | Trials per session *(optional)* |

---

## Result dict

| Key | Shape | Description |
|-----|-------|-------------|
| `params` | `(K, N)` | MAP parameter estimates per trial |
| `param_names` | `list[str]` | Parameter names |
| `hyper` | `dict` | Optimised hyperparameters |
| `log_evidence` | `float` | Log marginal likelihood |
| `hess_info` | `dict` | `W_std`: posterior std `(K, N)` |
| `duration` | `timedelta` | Wall-clock fitting time |

---

## GPU support

JAX automatically uses a GPU if one is available. Install the right backend first:

| Platform | Command |
|----------|---------|
| Apple Silicon (Metal) | `pip install jax-metal` |
| NVIDIA CUDA 12 | `pip install jax[cuda12]` |
| NVIDIA CUDA 11 | `pip install jax[cuda11_pip]` |

Then pass `device='gpu'` (or `'auto'`, the default) to `psytrax.fit()`.

---

## Web app

Visualise results and compare models interactively:

```bash
streamlit run app.py
```
