# Loading Datasets

psytrax does not require a custom file format. The fitting code expects an
in-memory dictionary, so the main job of a loader is to convert your lab's data
into trial-aligned arrays.

## Load a saved psytrax dictionary

Bundled examples use `.npy` files containing a dictionary:

```python
from pathlib import Path
import numpy as np

data = np.load(Path("data") / "DAP014_data.npy", allow_pickle=True).item()
```

This is convenient for examples and intermediate analysis files because it
preserves nested dictionaries such as `data["inputs"]`.

## Convert a CSV file

For tabular behavioural data, load the table with pandas and create the
dictionary explicitly:

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
```

Check that every trial-aligned array has the same length before fitting:

```python
n = len(data["responses"])
assert len(data["inputs"]["c"]) == n
assert len(data["times"]) == n
assert data["session_lengths"].sum() == n
```

## Load public IBL data

The repository includes a notebook that loads public International Brain
Laboratory behavioural data with ONE and converts it to psytrax format:

```text
examples/ibl_one_integration_walkthrough.ipynb
```

Use that notebook when you want a worked example of dataset search, session
selection, reaction-time reconstruction, and conversion to the psytrax data
dictionary.

## Keep loaders small

Good loaders should:

- Preserve one row per trial.
- Make response coding explicit.
- Store units in variable names or comments, especially reaction time.
- Keep private identifiers out of public examples.
- Return plain NumPy arrays and dictionaries so the result is easy to inspect.
