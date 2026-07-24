# psytrax Codex Instructions

psytrax is the submitted software toolbox for the NeurIPS response workflow. Treat
this repository as a production-facing package and Streamlit app, separate from
any sibling paper or response repository.

## Repository Map

- `psytrax/`: package source and model-fitting machinery.
- `app.py`: Streamlit app for fitting, visualisation, model comparison, recovery,
  and IBL exploration.
- `tests/`: fast unit and smoke tests.
- `docs/`: Sphinx documentation.
- `examples/`: code-first walkthroughs and figure-generation scripts.
- `data/`: bundled public/preprocessed example data. This directory is large and
  intentionally tracked.
- `example_fits/`: bundled example fit files used by the app and docs.
- `experiments/`: NeurIPS-response experiment scaffolding, manifests, Slurm
  templates, logs pointers, and exported results.

## Test Environment

Use the existing conda environment unless the user explicitly says otherwise:

```bash
conda run -n psytrax python -m pytest
```

For focused checks:

```bash
conda run -n psytrax python -m pytest tests/test_fit.py
conda run -n psytrax python -m pytest tests/test_models.py
```

If Streamlit app tests are added, run them with the same conda environment.

## Development Rules

- Do not push directly to `main`.
- `fit_all.py` must not commit or push results. Keep publishing and integration
  as explicit human-reviewed Git operations.
- Keep changes on review branches such as `neurips-response/setup`,
  `neurips-response/experiments`, `neurips-response/fixes`, or
  `neurips-response/streamlit-tests`.
- Keep commits focused and reviewable.
- Do not merge into `main` without explicit human approval.
- Do not overwrite or revert user changes unless the user explicitly requests it.
- Prefer small smoke tests before long fitting runs.

## Human Approval Gates

Stop and ask for explicit approval before:

- downloading any new dataset;
- accepting dataset licenses or terms;
- moving data into long-term or shared storage;
- launching full Slurm fitting runs;
- increasing sweep size, seed count, dataset count, model count, or walltime
  materially;
- using reviewer-requested datasets not already present in the workspace;
- pushing branches or tags to a remote;
- merging any branch into `main`;
- making manuscript or rebuttal claims based on new experimental results.

Approval must be scoped. Examples:

```text
APPROVED: download Dataset X from URL Y into path Z.
APPROVED: run experiment plan E-003 with max 40 Slurm jobs, 8 GPU-hours/job.
APPROVED: use result R-007 in the paper and rebuttal.
```

General approval to work on the NeurIPS response is not approval to download
datasets, run full Slurm jobs, push code, or make scientific claims.

## Slurm And Experiment Rules

- Before submitting Slurm jobs, create a dataset or experiment approval packet.
- Include reviewer concern, dataset provenance, license, expected size, storage
  path, preprocessing, fitting configs, Slurm resources, smoke test, expected
  outputs, and risks.
- Run local smoke tests before full Slurm sweeps when feasible.
- Use job arrays for parallel dataset/model/seed sweeps when appropriate.
- Save each submitted command, job ID, config, git commit, stdout/stderr path,
  and output artifact in an experiment manifest.
- Do not delete logs or intermediate results.
- Summarize failed and incomplete jobs separately from successful jobs.

## Data And Result Provenance

New large datasets, raw downloads, Slurm logs, caches, and exploratory fit files
should stay outside the main tracked package unless the user explicitly approves
adding them. Prefer tracked manifests, small summary tables, reproducible config
files, and final paper-ready figures.

Every reviewer-facing result should record:

- reviewer concern addressed;
- command or Slurm script used;
- psytrax git commit;
- dataset source and checksum when available;
- config path and hash when available;
- Slurm job IDs when relevant;
- output files;
- test or smoke-check status.

## Streamlit App Testing

When app behavior changes, add or update tests using Streamlit's `AppTest` where
practical. At minimum, cover:

- app imports and renders without exceptions;
- key pages or navigation paths render;
- bundled example fits load;
- critical widget interactions do not crash.

Before integration, also perform one real browser smoke test if the app UI or
visualisation behavior changed.

## Definition Of Done

A NeurIPS-response toolbox change is ready for human review when:

- relevant tests pass in the `psytrax` conda environment;
- new experiment scripts have smoke tests or dry-run paths;
- new results have provenance manifests;
- app changes have headless Streamlit tests where practical;
- generated artifacts are intentionally tracked or intentionally ignored;
- the final summary lists changed files, commands run, Slurm jobs submitted,
  result locations, unresolved risks, and required human decisions.
