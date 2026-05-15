# Data Format

psytrax fits trial-aligned arrays. The canonical input is a Python dictionary:

```python
data = {
    "inputs": {"c": contrast_array},
    "responses": response_array,
    "times": reaction_time_array,
    "session_lengths": trials_per_session,
}
```

## Required fields

| Key | Type | Description |
| --- | --- | --- |
| `inputs` | `dict[str, array]` | Trial-aligned task variables such as stimulus contrast, previous reward, or cue identity |
| `responses` | `array`, shape `(N,)` | Response variable, usually coded as 0/1 for binary choices |

## Optional fields

| Key | Alias accepted by `fit` | Description |
| --- | --- | --- |
| `times` | `T` | Reaction times, required by RT models such as the race and DDM models |
| `session_lengths` | `dayLength` | Number of trials in each session; needed for `session_boundaries=True` |
| `dopamine` | none | Optional per-trial dopamine peak used by the race model when dopamine hyperparameters are enabled |

## Model data requirements

Built-in models expose a `DATA_SPEC` dictionary. The web app uses this to help
map uploaded CSV columns to the fields a model needs. Custom models can expose
the same convention:

```python
DATA_SPEC = {
    "inputs": {
        "c": {"description": "Signed stimulus contrast", "required": True},
    },
    "response": {
        "key": "r",
        "description": "Choice, coded 0/1",
        "required": True,
    },
}
```

If a learning rule needs extra inputs, such as reward, add them under
`data["inputs"]`.
