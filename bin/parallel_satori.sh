#!/bin/sh           
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --exclusive


# Launch the code for parallel simulation:                                                                      
srun -n 512 /global/homes/z/zliu1997/Gkeyll/gkylsoft/gkyl/bin/gkyl M25_E2_3.lua restart
