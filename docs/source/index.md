# psytrax

Empirical Bayes fitting for trial-by-trial decision models.

psytrax helps researchers estimate how latent decision-making parameters change
over learning, fatigue, task engagement, or experimental manipulations. You
provide a per-trial likelihood; psytrax handles random-walk priors, MAP
fitting, Laplace evidence, model comparison, and uncertainty over trajectories.

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} {fas}`book;sd-text-primary` User guide
:link: user-guide/index
:link-type: doc

Installation, data format, dataset loading, and model structure.
:::

:::{grid-item-card} {fas}`chart-line;sd-text-primary` Examples
:link: examples/index
:link-type: doc

Worked analyses for fast choice models, RT models, recovery, and CSV data.
:::

:::{grid-item-card} {fas}`rocket;sd-text-primary` Web app
:link: https://psytrax.streamlit.app

Fit and inspect models in the browser, without writing code.
:::

:::{grid-item-card} {fas}`comments;sd-text-primary` Connect
:link: community/index
:link-type: doc

Ask questions, report issues, and help improve the project.
:::
::::

![](_static/images/psytrax_pipeline.svg)

## Overview

psytrax is designed for experiments where behaviour is organised one trial at a
time: choices, reaction times, stimuli, rewards, sessions, and other
trial-aligned signals. The package fits a Gaussian random-walk prior over model
parameters, so the output is a trajectory for each parameter rather than a
single static estimate.

Use psytrax when you want to:

- recover trial-by-trial changes in decision weights, biases, thresholds, or
  model-specific parameters;
- compare candidate behavioural models with approximate marginal likelihood;
- add learning rules or model-level hyperparameters without rewriting the
  inference engine;
- start from built-in logistic, DDM, or race models, then move to your own
  JAX-compatible per-trial likelihood.

```{toctree}
:hidden:
:maxdepth: 2
:caption: Documentation

tutorials/index
user-guide/index
examples/index
api/index
community/index
```
