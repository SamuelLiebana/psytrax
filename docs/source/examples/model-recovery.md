# Model Recovery

Model recovery is a sanity check: simulate data from known trajectories, fit the
model, and compare recovered parameters to the ground truth.

```python
import numpy as np
import psytrax
from psytrax.models import race

n_trials = 1000
true_params = np.stack(
    [
        np.linspace(1.0, 2.0, n_trials),
        np.linspace(1.0, 2.0, n_trials),
        np.full(n_trials, 0.5),
        np.full(n_trials, 0.5),
        np.linspace(1.0, 0.8, n_trials),
    ]
)

inputs = {
    "c": np.random.default_rng(0).choice(
        [-1.0, -0.5, 0.0, 0.5, 1.0],
        size=n_trials,
    )
}

result = psytrax.recover(
    sample_trial=race.sample_trial,
    log_lik_trial=race.log_lik_trial,
    n_params=race.N_PARAMS,
    true_params=true_params,
    inputs=inputs,
    param_names=race.PARAM_NAMES,
    true_model_hyper={"sig_i": 0.10},
)
```

Use recovery before trusting a new custom likelihood. If known trajectories
cannot be recovered from simulated data, the model may be weakly identifiable or
the initialisation may need work.
