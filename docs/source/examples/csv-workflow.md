# CSV Workflow

Many labs start with a CSV file containing one row per trial. Convert that table
into the psytrax dictionary explicitly so response coding and units are visible.

```python
import pandas as pd

trials = pd.read_csv("my_trials.csv")

data = {
    "inputs": {
        "c": trials["signed_contrast"].to_numpy(),
    },
    "responses": trials["choice_right"].to_numpy(),
    "times": trials["reaction_time_s"].to_numpy(),
    "session_lengths": trials.groupby("session_id").size().to_numpy(),
}

n_trials = len(data["responses"])
assert len(data["inputs"]["c"]) == n_trials
assert len(data["times"]) == n_trials
assert data["session_lengths"].sum() == n_trials
```

Once this assertion block passes, use the same `data` object with any built-in
model whose `DATA_SPEC` matches your columns.
