# Contributing

Contributions are most useful when they make psytrax easier to install, learn,
trust, or apply to new behavioural datasets.

## Good first contributions

- Try the first-fit tutorial on a clean environment and report confusing steps.
- Improve docstrings for public functions and model modules.
- Add small examples that answer one concrete user question.
- Add tests for edge cases in data validation, model likelihoods, or learning
  rules.
- Improve error messages when user data has the wrong shape or missing keys.

## Larger contributions

- Add a new model with `log_lik_trial`, `sample_trial`, `default_hyper`,
  `default_E0`, and `DATA_SPEC`.
- Add a dataset-loading guide for a public behavioural data source.
- Improve the Streamlit app workflow for non-coding users.
- Add model-recovery examples and benchmark datasets.
- Help set up documentation deployment and release automation.

## Contribution condition

Before opening a pull request, please make sure:

- The change has a clear user or maintainer benefit.
- New public behaviour is documented.
- New model behaviour includes tests or a model-recovery example.
- Large datasets, generated fits, and figures are not added unless they are
  intentionally part of the documented examples.
- The pull request is small enough to review in one sitting, or it is split
  into clearly staged pieces.

For substantial new functionality, open an issue first so the model assumptions,
API shape, and maintenance cost can be discussed before implementation.
