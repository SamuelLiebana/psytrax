# Examples

Examples show complete scientific workflows. Start with a small choice-only fit
if you are checking your installation, then move to reaction-time models, model
recovery, and data-loading examples.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Choice-only logistic fit
:link: logistic-first-fit
:link-type: doc
Fit a fast logistic baseline to contrast and choice data.
:::

:::{grid-item-card} Race model with reaction times
:link: race-model-rt
:link-type: doc
Fit a model that explains both choices and reaction times.
:::

:::{grid-item-card} Model recovery
:link: model-recovery
:link-type: doc
Simulate known trajectories and check whether psytrax can recover them.
:::

:::{grid-item-card} CSV workflow
:link: csv-workflow
:link-type: doc
Convert a trial table into the psytrax data dictionary.
:::
::::

## Existing scripts and notebooks

| Example | File | What it teaches |
| --- | --- | --- |
| Compare built-in models | `examples/compare_models_DAP009.py` | Fit several models to one mouse and compare evidence |
| IBL integration | `examples/ibl_one_integration_walkthrough.ipynb` | Load public IBL trials through ONE and convert them to psytrax format |
| Documentation figures | `examples/generate_dap014_docs_figures.py` | Generate trajectory and psychometric/chronometric figures from a saved race fit |

## App walkthrough media

Short GIFs are the next outreach asset to add. The most useful sequence would
show: upload or choose example data, run a model fit, inspect trajectories, and
compare model evidence. Keep each GIF focused on one task so it can be reused in
the README, talks, and social posts.

```{toctree}
:maxdepth: 1
:hidden:

logistic-first-fit
race-model-rt
model-recovery
csv-workflow
```
