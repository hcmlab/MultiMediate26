#!/bin/bash

# Submit all inference+plot jobs (step 3) for all datasets and test configs

# NoXi-base
sbatch run_3_test_plots.slurm configs/noxi-base/test/config1.json

# NoXi-J
sbatch run_3_test_plots.slurm configs/noxi-j/test/config1.json

# PInSoRo-CC
sbatch run_3_test_plots.slurm configs/pinsoro-cc/test/config1.json

# PInSoRo-CR
sbatch run_3_test_plots.slurm configs/pinsoro-cr/test/config1.json

# NoXi test-additional
sbatch run_3_test_plots.slurm configs/test-additional/test/config1.json

# MPII Group Interaction
sbatch run_3_test_plots.slurm configs/mpiigroupinteraction/test/config1.json
