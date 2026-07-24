# Slurm Result Harvest Prompt

```text
Harvest approved Slurm experiment results for psytrax.

Do not submit new jobs. Inspect the approved manifest, Slurm job IDs, stdout and
stderr logs, output files, and expected result paths. Separate successful,
failed, cancelled, timed-out, and still-running jobs. Do not fabricate missing
metrics.

Update or prepare a result summary with:
- experiment ID
- psytrax git commit
- config files
- dataset provenance
- Slurm job IDs
- command lines
- log paths
- output paths
- success/failure status
- metrics and confidence intervals when available
- files suitable for paper/rebuttal use
- remaining human decisions
```
