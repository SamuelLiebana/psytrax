#!/bin/bash
# Template only. Do not submit until a scoped experiment packet is approved.

#SBATCH --job-name=psytrax_E-XXX
#SBATCH --partition=<partition>
#SBATCH --array=0-0
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/logs/E-XXX_%A_%a.out
#SBATCH --error=experiments/logs/E-XXX_%A_%a.err

set -euo pipefail

EXPERIMENT_ID="E-XXX"
CONFIG_TABLE="experiments/manifests/${EXPERIMENT_ID}_jobs.tsv"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p experiments/logs experiments/results

echo "experiment_id=${EXPERIMENT_ID}"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID:-0}"
echo "submit_dir=${SLURM_SUBMIT_DIR}"
git rev-parse HEAD

# Expected TSV columns can be adapted per experiment. Keep column 1 as a shell
# command only if the file is generated and reviewed. Prefer explicit config
# paths plus a fixed runner command for production runs.
COMMAND=$(awk -F '\t' -v task="${SLURM_ARRAY_TASK_ID:-0}" 'NR == task + 2 {print $1}' "${CONFIG_TABLE}")

echo "command=${COMMAND}"
conda run -n psytrax bash -lc "${COMMAND}"
