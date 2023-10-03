#!/bin/sh       
#SBATCH --qos=debug     # debug or regular                                                                                           
#SBATCH -N 1          # Number of nodes                                                                         
#SBATCH -n 128         #Total Number of MPI tasks                                                                       
#SBATCH -t 00:30:00   # Time of simulation hours                                                    
#SBATCH --constraint=cpu    # CPU run                                                          
#SBATCH -J M25E10 # Name                                                               
# load up the correct modules, if required                                                                      
                                            
# Launch the code for parallel simulation:                                                                      
srun -n 128  /global/homes/z/zliu1997/gkylsoft/gkyl/bin/gkyl M25E10.lua 