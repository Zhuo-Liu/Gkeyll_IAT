#!/bin/sh
#SBATCH -N 1          # Number of nodes
#SBATCH -n 1         # Number of cores     
#SBATCH --qos psfc_24h
#SBATCH -t 24:00:00   # Time of simulation hours
#SBATCH -p sched_mit_psfc      # partition name
#SBATCH -J pgBex  # sensible name for the job
#SBATCH --mem 10000
# load up the correct modules, if required

module purge
#. /etc/profile.d/modules.sh
module use /home/software/psfc/modulefiles/
module load intel/2017-01
module load impi/2017-01
module load psfc/fftw/intel17/2.1.5
module load psfc/mkl/17
module load engaging/python/3.6.0
module load psfc/python/3.6-modules
module load engaging/idl
module load psfc/hdf5/intel-17/1.10.0
module load engaging/zlib/1.2.8
module load engaging/szip/2.1
module load psfc/adios/1.13.1
module load psfc/config

#module use /home/software/psfc/modulefiles/
#module load psfc/config
#module load engaging/python/3.6.0
#module load psfc/python/3.6-modules

# launch the code
python pgkylScript_2d_v7.py

