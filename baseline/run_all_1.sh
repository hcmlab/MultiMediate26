#!/bin/bash

# Submit all tuning jobs (step 1) for all datasets and configs

# NoXi-base
sbatch run_1_tuner_noxi.slurm configs/noxi-base/tune/config_audio_egemapsv2.json
sbatch run_1_tuner_noxi.slurm configs/noxi-base/tune/config_clip.json
sbatch run_1_tuner_noxi.slurm configs/noxi-base/tune/config_openface2.json
sbatch run_1_tuner_noxi.slurm configs/noxi-base/tune/config_openface3.json
sbatch run_1_tuner_noxi.slurm configs/noxi-base/tune/config_openpose.json
sbatch run_1_tuner_noxi.slurm configs/noxi-base/tune/config_audio_w2vbert2.json
sbatch run_1_tuner_noxi.slurm configs/noxi-base/tune/config_xlm_roberta.json
sbatch run_1_tuner_noxi.slurm configs/noxi-base/tune/config_dino.json
sbatch run_1_tuner_noxi.slurm configs/noxi-base/tune/config_swin.json
sbatch run_1_tuner_noxi.slurm configs/noxi-base/tune/config_videomae.json

# NoXi-J
sbatch run_1_tuner_noxi_j.slurm configs/noxi-j/tune/config_audio_egemapsv2.json
sbatch run_1_tuner_noxi_j.slurm configs/noxi-j/tune/config_clip.json
sbatch run_1_tuner_noxi_j.slurm configs/noxi-j/tune/config_openface2.json
sbatch run_1_tuner_noxi_j.slurm configs/noxi-j/tune/config_openface3.json
sbatch run_1_tuner_noxi_j.slurm configs/noxi-j/tune/config_openpose.json
sbatch run_1_tuner_noxi_j.slurm configs/noxi-j/tune/config_audio_w2vbert2.json
sbatch run_1_tuner_noxi_j.slurm configs/noxi-j/tune/config_xlm_roberta.json
sbatch run_1_tuner_noxi_j.slurm configs/noxi-j/tune/config_dino.json
sbatch run_1_tuner_noxi_j.slurm configs/noxi-j/tune/config_swin.json
sbatch run_1_tuner_noxi_j.slurm configs/noxi-j/tune/config_videomae.json

# PInSoRo-CC
sbatch run_1_tuner_pinsoro_cc.slurm configs/pinsoro-cc/tune/config_audio_egemapsv2.json
sbatch run_1_tuner_pinsoro_cc.slurm configs/pinsoro-cc/tune/config_clip.json
sbatch run_1_tuner_pinsoro_cc.slurm configs/pinsoro-cc/tune/config_openface2.json
sbatch run_1_tuner_pinsoro_cc.slurm configs/pinsoro-cc/tune/config_openface3.json
sbatch run_1_tuner_pinsoro_cc.slurm configs/pinsoro-cc/tune/config_openpose.json
sbatch run_1_tuner_pinsoro_cc.slurm configs/pinsoro-cc/tune/config_audio_w2vbert2.json
sbatch run_1_tuner_pinsoro_cc.slurm configs/pinsoro-cc/tune/config_xlm_roberta.json
sbatch run_1_tuner_pinsoro_cc.slurm configs/pinsoro-cc/tune/config_dino.json
sbatch run_1_tuner_pinsoro_cc.slurm configs/pinsoro-cc/tune/config_swin.json
sbatch run_1_tuner_pinsoro_cc_videomae.slurm configs/pinsoro-cc/tune/config_videomae.json



# PInSoRo-CR
sbatch run_1_tuner_pinsoro_cr.slurm configs/pinsoro-cr/tune/config_audio_egemapsv2.json
sbatch run_1_tuner_pinsoro_cr.slurm configs/pinsoro-cr/tune/config_clip.json
sbatch run_1_tuner_pinsoro_cr.slurm configs/pinsoro-cr/tune/config_openface2.json
sbatch run_1_tuner_pinsoro_cr.slurm configs/pinsoro-cr/tune/config_openface3.json
sbatch run_1_tuner_pinsoro_cr.slurm configs/pinsoro-cr/tune/config_openpose.json
sbatch run_1_tuner_pinsoro_cr.slurm configs/pinsoro-cr/tune/config_audio_w2vbert2.json
sbatch run_1_tuner_pinsoro_cr.slurm configs/pinsoro-cr/tune/config_xlm_roberta.json
sbatch run_1_tuner_pinsoro_cr.slurm configs/pinsoro-cr/tune/config_dino.json
sbatch run_1_tuner_pinsoro_cr.slurm configs/pinsoro-cr/tune/config_swin.json
sbatch run_1_tuner_pinsoro_cr.slurm configs/pinsoro-cr/tune/config_videomae.json
