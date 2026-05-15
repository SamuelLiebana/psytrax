# psytrax

Empirical Bayes fitting for trial-by-trial decision models.

psytrax is for researchers who have behavioural data organised by trial and
want to estimate how latent decision-making parameters change over learning,
fatigue, task engagement, or experimental manipulations. You provide a
per-trial likelihood; psytrax handles random-walk priors, MAP estimation,
Laplace evidence, model comparison, and uncertainty over trajectories.

The documentation is organised around the
[Diataxis framework](https://diataxis.fr/): tutorials for learning by doing,
how-to material for common tasks, reference material for precise API details,
and explanation pages for the modelling ideas.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Start with a real fit
:link: tutorials/first-fit
:link-type: doc
Follow a narrative walkthrough for a new experimentalist joining Ann's lab.
:::

:::{grid-item-card} Prepare your data
:link: user-guide/data-format
:link-type: doc
Learn the input dictionary format and how model data requirements are declared.
:::

:::{grid-item-card} Choose or write a model
:link: user-guide/models
:link-type: doc
Use built-in logistic, DDM, race, and MLP models, or supply your own likelihood.
:::

:::{grid-item-card} Join the project
:link: community/index
:link-type: doc
Find support routes, contribution priorities, and the public roadmap.
:::
::::

```{toctree}
:maxdepth: 2
:caption: Documentation

tutorials/index
user-guide/index
examples/index
api/index
community/index
maintenance
publication
```
