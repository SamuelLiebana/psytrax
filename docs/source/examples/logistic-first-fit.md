# Choice-Only Logistic Fit

Use the logistic model as a fast first check that your data are shaped
correctly. It only needs stimulus inputs and choices, so it is a good baseline
before fitting reaction-time models.

```python
import numpy as np
import psytrax
from psytrax.models import logistic

data = np.load("data/DAP014_data.npy", allow_pickle=True).item()

choice_data = {
    "inputs": {"c": data["inputs"]["c"]},
    "responses": data["responses"],
    "session_lengths": data["session_lengths"],
}

n_trials = len(choice_data["responses"])

result = psytrax.fit(
    data=choice_data,
    log_lik_trial=logistic.log_lik_trial,
    n_params=logistic.N_PARAMS,
    param_names=logistic.PARAM_NAMES,
    hyper=logistic.default_hyper(),
    E0=logistic.default_E0(n_trials),
    session_boundaries=True,
)
```

The logistic model returns two trajectories by default: contrast weight `w` and
choice bias `b`. Use this fit to verify trial alignment and to establish a
choice-only evidence baseline.
