# psytrax

**Empirical Bayes fitting for trial-by-trial decision models**

psytrax fits a Gaussian random-walk prior over a sequence of K parameters across N trials and optimises the prior variance (hyperparameters) by maximising the marginal likelihood (evidence) using the Laplace approximation.

It is model-agnostic: you supply a **per-trial log-likelihood function** and psytrax handles all the inference machinery.

---

## Installation

For the web app or general use (installs all dependencies including Streamlit):

```bash
pip install -r requirements.txt
```

For Streamlit Community Cloud deployment, keep `requirements.txt` as the only
recognized Python dependency file in the repo root. Community Cloud prioritizes
`environment.yml` over `requirements.txt` and will ignore the latter if both are
present.

For development and tests only:

```bash
pip install -e .[dev]
```

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

If you want a single process-noise hyperparameter shared across all parameters,
pass `shared_sigma=True`:

```python
result = psytrax.fit(
    data=data,
    log_lik_trial=log_lik_trial,
    n_params=N_PARAMS,
    shared_sigma=True,
)
```

The built-in models expose the same option through `default_hyper(shared_sigma=True)`.

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
| DDM (exact) | `models/ddm.py` | 4 | Yes | Drift diffusion model — Navarro & Fuss (2009) / Bogacz et al. (2006) series solution |
| DDM (approx) | `models/ddm_approx.py` | 3 | Yes | Drift diffusion model — inverse-Gaussian single-barrier approximation |
| Race | `models/race.py` | 6 | Yes | Race model with separate accumulators |
| MLP | `models/mlp.py` | 13 | No | 1→4→1 MLP with tanh hidden layer |

Each model exposes: `log_lik_trial`, `N_PARAMS`, `PARAM_NAMES`, `default_hyper()`, `default_E0(N)`.
For a shared random-walk variance across parameters, call `default_hyper(shared_sigma=True)`.

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

psytrax requires float64 precision for stable Hessian computation and Laplace evidence. On Apple Silicon, psytrax uses a **hybrid** path when Metal is available: Metal float32 for MAP optimisation, then CPU float64 for Hessian / evidence. NVIDIA CUDA supports float64 directly and will accelerate fitting:

| Platform | Command |
|----------|---------|
| NVIDIA CUDA 12 | `pip install jax[cuda12]` |
| NVIDIA CUDA 11 | `pip install jax[cuda11_pip]` |

For most users, `device='auto'` is the recommended default.

---

## Performance

Wall-clock fitting times on Apple M4 CPU (JAX L-BFGS, float64), measured on bundled example mice spanning lower and higher trial counts. Times include warm-start, hyperparameter optimisation, and Hessian computation.

| Model | **DAP044** (520) | **DAP027** (2,535) | **DAP009** (5,289) | **DAP011** (5,681) | **DAP031** (7,836) | **DAP007** (11,411) |
|---|---|---|---|---|---|---|
| Logistic (K=2) | 2.1s | 10.8s | 10.3s | 23.0s | 19.1s | 20.3s |
| DDM approx (K=3) | 5.3s | 11.1s | 28.4s | 45.6s | 40.5s | 35.8s |
| Race fixed sig_i (K=5) | 12.6s | 28.2s | 13.9s | 76.9s | 72.8s | 101.2s |

These timings are intentionally based on real datasets rather than synthetic sweeps, so they are not perfectly monotonic in trial count. Fit time depends not just on `N`, but also on how many hyperparameter cycles the data trigger before the log-evidence stops improving.

NVIDIA CUDA (float64) is expected to give a further **3–8× speedup** for models with K ≥ 3, since the per-trial likelihood and the entire MAP loop run on-device via `jax.vmap`.

---

## Web app

The web app lets you fit models, visualise results, and compare models — all from a browser, with no coding required.

### Option A — hosted (zero install)

The app is deployed on Streamlit Community Cloud. Open the link and use it directly:

> **[https://psytrax.streamlit.app](https://psytrax.streamlit.app)**

Fitting on the cloud is slower than running locally, and uploaded files are not persisted between sessions.

### Option B — run locally

**Requirements:** Python ≥ 3.10, [conda](https://docs.anaconda.com/miniconda/) recommended.

```bash
# 1. Clone the repository
git clone https://github.com/SamuelLiebana/psytrax.git
cd psytrax

# 2. Create and activate a virtual environment
conda create -n psytrax python=3.11
conda activate psytrax

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

**GPU acceleration (optional)**

| Platform | Extra install |
|----------|---------------|
| Apple Silicon (Metal) | `pip install jax-metal` |
| NVIDIA CUDA 12 | `pip install jax[cuda12]` |

### Pages

| Page | What it does |
|------|-------------|
| Instructions | Usage guide |
| Fit Model | Upload a dataset (`.npy` or `.csv`), choose a model, run the fit, download results |
| Visualise Results | Load a saved fit and explore trial-by-trial parameter trajectories |
| Compare Models | Overlay multiple fits and compare log-evidence scores |
