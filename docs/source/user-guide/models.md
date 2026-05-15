# Models

psytrax is model-agnostic: a model is mainly a per-trial log-likelihood written
with JAX-compatible operations. Built-in models provide ready-made likelihoods,
default hyperparameters, initial values, simulators, and data specifications.

## Built-in models

| Model | Module | Choice | RT | Typical first use |
| --- | --- | --- | --- | --- |
| Logistic | `psytrax.models.logistic` | yes | no | Fast choice-only baseline |
| DDM | `psytrax.models.ddm` | yes | yes | Exact diffusion-model likelihood |
| Race | `psytrax.models.race` | yes | yes | Choice and RT fits with optional dopamine term |

## A minimal custom likelihood

```python
import jax
import jax.numpy as jnp


def log_lik_trial(params, dat_trial, model_hyper=None):
    w, b = params
    x = dat_trial["inputs"]["x"]
    y = dat_trial["r"]
    logit = w * x + b
    return y * jax.nn.log_sigmoid(logit) + (1 - y) * jax.nn.log_sigmoid(-logit)
```

The likelihood receives scalar trial data because psytrax vectorises across
trials internally. Use `jax.numpy`, not regular `numpy`, inside the likelihood
so gradients and Hessians can be computed.

## Suggested model workflow

1. Fit a simple logistic model as a behavioural baseline.
2. Fit an RT model if reaction time is central to the question.
3. Compare models with log evidence on the same trials.
4. Run model recovery on synthetic data before trusting a new likelihood.
5. Document any task-specific assumptions in the model's docstring and
   `DATA_SPEC`.
