# Contributing

Contributions are most useful when they make psytrax easier to install, learn,
trust, or apply to real behavioural datasets. This can mean code, documentation,
examples, testing, user feedback, or careful bug reports.

## Before you start

Please check the [issue tracker](https://github.com/SamuelLiebana/psytrax/issues)
before starting substantial work.

- If an issue already exists, comment there so others know you are interested.
- If no issue exists, open one describing the idea, the scientific use case, and
  the likely scope.
- For exploratory ideas, start in Zulip once the public psytrax stream is live.

Some issues may be aspirational rather than ready to implement. An issue is
ready to work on when the expected behaviour, user benefit, and likely review
scope are clear enough that a maintainer can evaluate the result.

Good first contributions include:

- trying tutorials in a clean environment and reporting confusing steps;
- improving public docstrings and API reference pages;
- adding small examples that answer one concrete scientific workflow question;
- improving error messages for malformed data;
- adding tests for data validation, model likelihoods, or learning rules.

## Development setup

Clone the repository and create an editable install:

```bash
git clone https://github.com/SamuelLiebana/psytrax.git
cd psytrax
conda create -n psytrax-dev python=3.10
conda activate psytrax-dev
python -m pip install -e ".[dev,docs]"
```

If you already have the `psytrax` conda environment used for analysis, you can
install the editable package and docs extras there instead:

```bash
conda activate psytrax
python -m pip install -e ".[dev,docs]"
```

## Running tests

Run the full test suite from the repository root:

```bash
python -m pytest
```

For narrow changes, it is fine to run a focused test first:

```bash
python -m pytest tests/test_fit.py
python -m pytest tests/test_models.py
```

All new model behaviour, learning-rule behaviour, or validation behaviour should
come with tests. If a change affects numerical fitting, include a small smoke
test rather than a long-running full analysis.

## Building documentation

The documentation is built with Sphinx:

```bash
sphinx-build -W -b html docs/source docs/build/html
```

The `-W` flag treats warnings as errors. Please use it before opening a pull
request because the GitHub Actions docs workflow uses the same strict build.

When adding a new documentation page:

- add the page to the appropriate `toctree`;
- include enough context for a new user to know when to use it;
- prefer runnable snippets over pseudocode where practical;
- keep private lab data, identifiable data, and large generated outputs out of
  the repository.

## Adding examples

Examples should teach a complete workflow, not just a function call. A good
psytrax example usually states:

- what scientific question the example answers;
- what data fields are required;
- which model is fit and why;
- how to interpret `params`, `hyper`, and `log_evidence`;
- what a user should try next with their own data.

For small examples, write a docs page under `docs/source/examples/`. For longer
or executable analyses, add a script or notebook under `examples/` and link to
it from the examples gallery.

## Adding or changing models

New model contributions should include:

- `log_lik_trial(params, dat_trial, model_hyper=None)`;
- `sample_trial(...)` when simulation or model recovery is possible;
- `N_PARAMS` and `PARAM_NAMES`;
- `default_hyper(...)` and `default_E0(N)`;
- `DATA_SPEC` so the web app and docs can describe required columns;
- tests covering valid data, invalid data, and finite log likelihoods;
- a short documentation example or model-recovery example.

Model assumptions should be explicit in the module docstring. If a parameter is
not identifiable, constrained, transformed, or only valid in a particular data
regime, say so.

## Data contributions

Small public example datasets are welcome when they make the documentation more
useful. Before adding data:

- confirm the data can be shared publicly;
- remove private identifiers and sensitive metadata;
- keep files small enough for GitHub;
- document units, response coding, and session structure;
- prefer synthetic data when the example does not need real data.

Large raw datasets should not be committed to the repository. Link to an
external archive and provide a lightweight loader instead.

## Pull request checklist

Before opening a pull request, check that:

- the change has a clear user or maintainer benefit;
- tests pass locally with `python -m pytest`;
- docs build locally with `sphinx-build -W -b html docs/source docs/build/html`;
- public behaviour is documented;
- new examples are linked from the examples page;
- generated files, large outputs, and local build artifacts are not committed;
- the pull request description explains the motivation and links related issues.

Draft pull requests are welcome. They are useful when you want feedback on API
shape, model assumptions, or documentation direction before the work is final.

## Review expectations

Maintainers will prioritise correctness, clarity, and long-term usability. For
scientific code, expect review questions about data assumptions, parameter
identifiability, numerical stability, tests, and documentation. For user-facing
docs, expect review questions about whether a new student or experimentalist
could follow the workflow without private context.
