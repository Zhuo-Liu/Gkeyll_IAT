#!/bin/sh           
#SBATCH --qos=regular
#SBATCH --time=48:00:00
#SBATCH -N 16 
#SBATCH -n 512
#SBATCH -J 25_3 # sensible name for the job                
#SBATCH --constraint=haswell
# load up the correct modules, if required                                                                      

# Launch the code for parallel simulation:                                                                      
srun -n 512 /global/homes/z/zliu1997/Gkeyll/gkylsoft/gkyl/bin/gkyl M25_E2_3.lua restart
