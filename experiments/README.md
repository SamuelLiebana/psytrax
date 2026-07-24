# NeurIPS Response Experiments

This directory holds review-response experiment scaffolding for psytrax. Keep
large raw datasets, cache directories, full Slurm logs, and exploratory fit
artifacts outside tracked source unless the user explicitly approves adding
them.

## Layout

- `slurm/`: Slurm templates and dry-run helpers.
- `manifests/`: tracked experiment manifests and approval packets.
- `results/`: small, paper-ready summaries, tables, and figures.

## Workflow

1. Create an approval packet in `manifests/` before new data downloads or full
   fitting runs.
2. Wait for explicit human approval.
3. Run a local smoke test with the `psytrax` conda environment.
4. Submit approved Slurm jobs.
5. Record job IDs, commands, logs, configs, data provenance, and outputs in a
   manifest.
6. Export only reviewer-facing summaries to `results/`.

## Local Baseline

```bash
conda run -n psytrax python -m pytest
```

## Result Naming

Prefer stable experiment IDs:

```text
E-001-short-name
E-002-short-name
```

Use the same ID in approval packets, Slurm job names, logs, manifests, and
result files.
