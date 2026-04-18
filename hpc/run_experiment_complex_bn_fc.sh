#!/bin/bash -l
#SBATCH -J cnn_opt_complex # Job name
#SBATCH -N 1 # how many nodes

#SBATCH -A plgreforming-cpu  # zespol-zasoby

#SBATCH -p plgrid #queue name
#SBATCH --time=72:00:00

#SBATCH --cpus-per-task=24 # cores per task
#SBATCH --output="opt_complex_bn_fc.out" # screen output file
#SBATCH --error="opt_complex_bn_fc.err"


cd $SLURM_SUBMIT_DIR
module add .plgrid
module add plgrid/apps/matlab
module add plgrid/apps/matlab/R2024b

matlab -batch "run_main_optimization"


