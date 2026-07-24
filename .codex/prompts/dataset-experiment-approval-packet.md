# Dataset/Experiment Approval Packet Template

Use this template before downloading new datasets, preprocessing new data, or
launching full Slurm fitting runs.

```md
# Approval Packet: E-XXX

## Reviewer Concern

- Reviewer:
- Concern:
- Why this experiment addresses it:

## Dataset

- Dataset name:
- Source URL or access route:
- Citation:
- License or terms:
- Expected size:
- Proposed storage path:
- Privacy or sensitivity notes:
- Checksum plan:

## Preprocessing

- Script or command:
- Inputs:
- Outputs:
- Exclusions or filtering:
- Reproducibility notes:

## Fit Plan

- Model(s):
- Dataset split(s):
- Seeds:
- Number of fits:
- Smoke-test command:
- Full-run command or Slurm script:

## Slurm Resources

- Partition:
- CPUs per task:
- GPUs:
- Memory:
- Walltime:
- Job array size:
- Expected total compute:
- Log paths:

## Outputs

- Manifest path:
- Result path:
- Paper-ready table or figure path:
- Success criteria:

## Risks

- Licensing:
- Runtime:
- Storage:
- Numerical stability:
- Reviewer-claim risk:

## Approval Needed

APPROVED: ...
```
