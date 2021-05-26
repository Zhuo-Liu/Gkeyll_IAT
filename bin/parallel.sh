#!/bin/sh                                                                                                       
#SBATCH -N 8          # Number of nodes                                                                         
#SBATCH -n 256         # Number of cores                                                                          
#SBATCH -t 24:00:00   # Time of simulation hours 
#SBATCH --qos psfc_24h                                                               
#SBATCH -p sched_mit_psfc    # partition name                                                             
#SBATCH -J Biskamp # sensible name for the job                                                                   
# load up the correct modules, if required                                                                      
                                            
# Launch the code for parallel simulation:                                                                      
mpirun -n 256 /home/zhuol/gkylsoft/gkyl/bin/gkyl Biskamp_massRatio400_Tratio50.lua