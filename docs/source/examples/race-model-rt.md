# Race Model With Reaction Times

The race model uses choices and reaction times. It is useful when you want to
separate changes in stimulus weights, choice baselines, and decision threshold.

```python
import numpy as np
import psytrax
from psytrax.models import race

data = np.load("data/DAP014_data.npy", allow_pickle=True).item()
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
)
```

Inspect `result["params"]` to see the fitted trajectories for `wr`, `wl`,
`br`, `bl`, and `z`. Compare `result["log_evidence"]` against simpler models
fit to the same trials.
