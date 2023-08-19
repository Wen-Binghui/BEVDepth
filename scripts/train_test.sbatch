#!/bin/bash
#SBATCH --job-name="nuScenes_ext"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4,VRAM:20G
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user="wwenbinghui@126.com"
#SBATCH --output=/home/wiss/lhao/binghui_DONTDELETE_ME/thesis/BEVDepth/slurm_log/slurm-%j-output.out
#SBATCH --error=/home/wiss/lhao/binghui_DONTDELETE_ME/thesis/BEVDepth/slurm_log/slurm-%j-error.out
#SBATCH --partition=NORMAL
#SBATCH --comment="andy"

export GROUP_NAME_ANDY=bev_depth_lss_r50_256x704_128x128_24e_2key_testtest
/usr/wiss/lhao/anaconda3/envs/andy-mmcv-torch1.11.0/bin/python bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py --amp_backend native -b 8 --gpus 4