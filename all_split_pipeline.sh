#!/bin/bash
set -euo pipefail

preprocess_job_id="$(sbatch --parsable preprocess.slurm)"
preprocess_job_id="${preprocess_job_id%%;*}"
echo "Submitted preprocess job ${preprocess_job_id}"

sbatch --dependency="afterok:${preprocess_job_id}" boltz.slurm
