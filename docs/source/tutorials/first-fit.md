# Fit your first behavioural dataset

Imagine you have just joined Ann's lab as a master's student. You have a file
with one row per trial: stimulus contrast, the animal's choice, reaction time,
and session boundaries. Your first goal is not to become an expert in empirical
Bayes. It is to answer a concrete question:

> Do the decision parameters change over the experiment, and can those changes
> explain the choices and reaction times?

This tutorial uses one bundled example mouse (`DAP014`) and the built-in race
model. The same workflow applies to your own data once it has the same basic
shape.

## 1. Install psytrax

For a local analysis environment:

```bash
python -m pip install -e ".[dev]"
```

For a lighter package install without the web app:

```bash
python -m pip install -e .
```

## 2. Load the example data

```python
from pathlib import Path
import numpy as np

data_path = Path("data") / "DAP014_data.npy"
data = np.load(data_path, allow_pickle=True).item()

print(data.keys())
print(data["inputs"].keys())
print(data["responses"].shape)
```

The important fields are:

| Field | Meaning |
| --- | --- |
| `inputs["c"]` | Signed stimulus contrast for each trial |
| `responses` | Choice on each trial, coded as 0/1 |
| `times` | Reaction time on each trial |
| `session_lengths` | Number of trials in each behavioural session |

## 3. Pick a model

The race model predicts both choices and reaction times. It has five
trial-varying parameters:

```python
from psytrax.models import race

print(race.PARAM_NAMES)
```

Those names correspond to right and left stimulus weights (`wr`, `wl`), right
and left baselines (`br`, `bl`), and a decision threshold (`z`).

## 4. Run a first fit

Start with a smaller slice while you are checking the pipeline. This gives you
fast feedback if your data format is wrong.

```python
import psytrax

n_trials = 500

result = psytrax.fit(
    data=data,
    log_lik_trial=race.log_lik_trial,
    n_params=race.N_PARAMS,
    param_names=race.PARAM_NAMES,
    hyper=race.default_hyper(),
    E0=race.default_E0(n_trials),
    model_hyper=race.default_model_hyper(),
    n_trials=n_trials,
    verbose=True,
)

print(result["params"].shape)
print(result["log_evidence"])
print(result["hyper"])
```

`result["params"]` has shape `(K, N)`: one trajectory per model parameter.
The log evidence is useful when comparing models fit to the same data.

## 5. Plot the trajectories

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(race.N_PARAMS, 1, sharex=True, figsize=(8, 7))

for ax, name, values in zip(axes, result["param_names"], result["params"]):
    ax.plot(values)
    ax.set_ylabel(name)

axes[-1].set_xlabel("Trial")
fig.tight_layout()
plt.show()
```

At this point you should ask domain questions rather than software questions:

- Do the trajectories change around session starts?
- Do weights become more asymmetric as the animal learns?
- Does the threshold increase during slow or cautious periods?
- Does a simpler choice-only model explain the data nearly as well?

## 6. Scale up carefully

Once the first 500 trials work, remove `n_trials` and include session
boundaries:

```python
n_trials = len(data["responses"])

result = psytrax.fit(
    data=data,
    log_lik_trial=race.log_lik_trial,
    n_params=race.N_PARAMS,
    param_names=race.PARAM_NAMES,
    hyper=race.default_hyper(),
    E0=race.default_E0(n_trials),
    model_hyper=race.default_model_hyper(),
    session_boundaries=True,
    verbose=True,
)
```

This full fit estimates a larger process noise at session boundaries, which is
often important for multi-day behavioural experiments.

## 7. What to do with your own data

For a new dataset, make a dictionary with the same top-level structure:

```python
data = {
    "inputs": {"c": contrast},
    "responses": choices,
    "times": reaction_times,
    "session_lengths": trials_per_session,
}
```

All trial-aligned arrays should have the same length. If your task has more
features, start with the logistic model's `make_model(...)` helper or write a
custom `log_lik_trial` function.
