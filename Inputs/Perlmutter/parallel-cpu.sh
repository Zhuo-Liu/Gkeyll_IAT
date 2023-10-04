#!/bin/sh       
#SBATCH --qos=regular     # debug or regular                                                                                           
#SBATCH -N 1          # Number of nodes                                                                         
#SBATCH -n 128         #Total Number of MPI tasks                                                                       
#SBATCH -t 24:00:00   # Time of simulation hours                                                    
#SBATCH --constraint=cpu    # CPU run                                                          
#SBATCH -J M25E10_old # Name                                                               
# load up the correct modules, if required                                                                      
                                            
# Launch the code for parallel simulation:                                                                      
srun -n 128 /pscratch/sd/z/zliu1997/IAT/mass25/M25E10/gkyl M25E10.lua 