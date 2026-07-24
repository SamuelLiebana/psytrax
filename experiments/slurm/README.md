# Slurm Templates

Templates in this directory are examples only. Do not submit full Slurm jobs
until the user has approved a scoped experiment packet.

Recommended progression:

1. Fill in an approval packet under `experiments/manifests/`.
2. Run a local smoke test.
3. Render or copy an approved Slurm script.
4. Submit with `sbatch`.
5. Record job IDs and log paths in the manifest.

Typical monitoring commands:

```bash
squeue -u "$USER"
sacct -j "<job_id>" --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
```
