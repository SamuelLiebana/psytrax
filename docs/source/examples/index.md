# Examples

Examples show complete scientific workflows. Start with a small choice-only fit
if you are checking your installation, then move to reaction-time models, model
recovery, and data-loading examples.

<div class="psytrax-example-gallery">
  <figure class="psytrax-example-wide">
    <img src="../_static/images/examples/long-term-learning-task.png" alt="Long-term learning behavioural task timeline with stimulus onset, auditory go cue, choice, stimulus centre, reward, and stimulus contrast values." />
    <figcaption><strong>Behavioural task.</strong> A long-term learning task with stimulus contrast, go cue, choice, stimulus centering, and reward events aligned within each trial.</figcaption>
  </figure>
  <figure>
    <img src="../_static/images/examples/shallow-logistic-schematic.png" alt="Schematic of a shallow logistic model that maps contrast and bias to response probability." />
    <figcaption><strong>Choice model.</strong> A simple logistic model is a useful first fit before moving to richer reaction-time models.</figcaption>
  </figure>
  <figure>
    <img src="../_static/images/examples/dap014-race-trajectories.png" alt="Race-model parameter trajectories for DAP014." />
    <figcaption><strong>Parameter trajectories.</strong> A saved race-model fit showing trial-by-trial parameter estimates and uncertainty.</figcaption>
  </figure>
  <figure>
    <img src="../_static/images/examples/dap014-race-psychometric.png" alt="Psychometric curves evolving over learning for DAP014." />
    <figcaption><strong>Psychometric evolution.</strong> Fitted choice predictions summarised across learning windows.</figcaption>
  </figure>
  <figure>
    <img src="../_static/images/examples/dap014-race-chronometric.png" alt="Chronometric curves evolving over learning for DAP014." />
    <figcaption><strong>Chronometric evolution.</strong> Reaction-time predictions from the same race-model fit.</figcaption>
  </figure>
</div>

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

```{toctree}
:maxdepth: 1
:hidden:

logistic-first-fit
race-model-rt
model-recovery
csv-workflow
```
