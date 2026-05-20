#!/bin/bash

# Submit all fairness evaluation jobs (step 4) for all datasets

# NoXi-base
sbatch run_4_test_fairness_allmod_psess.slurm configs/noxi-base/test/config1.json

# NoXi-J
sbatch run_4_test_fairness_allmod_psess.slurm configs/noxi-j/test/config1.json

# PInSoRo-CC
sbatch run_4_test_fairness_allmod_psess.slurm configs/pinsoro-cc/test/config1.json

# PInSoRo-CR
sbatch run_4_test_fairness_allmod_psess.slurm configs/pinsoro-cr/test/config1.json

# NoXi test-additional
sbatch run_4_test_fairness_allmod_psess.slurm configs/test-additional/test/config1.json

# MPII Group Interaction
sbatch run_4_test_fairness_allmod_psess.slurm configs/mpiigroupinteraction/test/config1.json
