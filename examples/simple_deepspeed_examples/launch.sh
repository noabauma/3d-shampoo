#!/bin/bash
# Run the examples on all locally available GPUs.
#
# On a SLURM cluster, submit the same commands with srun instead
# (see the git history for the original Piz Daint job script).

set -e

# data parallelism only
deepspeed ds_no_pp.py --deepspeed_config ds_config.json

# pipeline parallelism; with N GPUs and S stages you get N/S data-parallel replicas
deepspeed ds_pp.py --deepspeed_config ds_config.json --stages 1
