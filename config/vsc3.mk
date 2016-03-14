
ifeq ($(strip $(HPC_SYSTEM)),vsc3)
INCLUDE += -I$(HOME)/include
INCLUDE += -I/cm/shared/apps/intel/impi_5.0.3/intel64/include
INCLUDE += -I/opt/sw/x86_64/glibc-2.12/ivybridge-ep/netcdf/4.3.2/intel-14.0.2/include
INCLUDE += -I/opt/sw/x86_64/glibc-2.12/ivybridge-ep/hdf5/1.8.12/intel-14.0.2/include
GLFLAGS  = -lm
CC=icc
MPICC=mpiicc
OPT=-O3 -xHost
MPICFLAGS+= -DMPICH_IGNORE_CXX_SEEK
OMPFLAG=-openmp
LIBS    +=-L/opt/sw/x86_64/glibc-2.12/ivybridge-ep/hdf5/1.8.12/intel-14.0.2/lib -lhdf5 -lhdf5_hl
LIBS    +=-L/cm/shared/apps/intel-cluster-studio/composer_xe_2013_sp1.2.144/compiler/lib/intel64 -lirc -lsvml
LIBS    += -L/cm/shared/apps/intel/impi_5.0.3/intel64/lib -lmpi
LIBS    +=-L/opt/sw/x86_64/glibc-2.12/ivybridge-ep/netcdf/4.3.2/intel-14.0.2/lib -lnetcdf -lcurl
endif
#############################modules to load in .bashrc#######################
#module load intel/15.0.2
#module load intel-mpi/5
#module load hdf5/1.8.12
#module load netcdf/4.3.2
####################sample submit script for mpi+omp job##########################
##! /bin/bash
#
#
##SBATCH -J benchmark
##SBATCH --output=more_nodes.txt
##SBATCH -N32                 # number of nodes
##SBATCH --ntasks-per-node=2 # mpi processes on each node
##SBATCH --ntasks-per-core=1 # no hyperthreading
##SBATCH --cpus-per-task=8   # openMP num threads
##SBATCH --time=02:00:00
#
#
#export I_MPI_PIN=yes
#export I_MPI_PIN_DOMAIN=omp:platform
#export I_MPI_PIN_CELL=core
#
#
#make cluster_mpib device=omp
#echo "# npx npy npz #procs #threads n Nx Ny Nz t_AXPBY t_DOT t_DX t_DY t_DZ t_ARAKAWA #iterations t_1xELLIPTIC_CG t_DS" > benchmark.dat
#export OMP_NUM_THREADS=8
#echo 1 2 32 3 200 200 32 | mpirun -n 64 ./cluster_mpib >> benchmark.dat 2>&1
############OR##############################################################
#module load intel-mpi/5 intel/14.02 likwid/4.0
#module load hdf5/1.8.12
#module load netcdf/4.3.2
#########################################Submit with LIKWID PIN###############
##!/bin/bash
## submit with sbatch 
#
##SBATCH -J fq11c24t10a10z03s05_FELTOR
##SBATCH --output=fq11c24t10a10z03s05.out
##SBATCH -N 14 
####SBATCH --ntasks-per-node=2
####SBATCH --ntasks-per-core=1
####SBATCH --cpus-per-task=8
#
#export I_MPI_PMI_LIBRARY=/cm/shared/apps/slurm/current/lib/libpmi.so
#
#export FILE=fq11c24t10a10z03s05
#export SCRATCH=/home/lv70607/mwiesenberge
#export NAME=n3N180N21
#
#module purge
#module load intel-mpi/5 intel/14.0.2 likwid/4.0
#module load hdf5/1.8.12
#module load netcdf/4.3.2
#
#mkdir -p $SCRATCH/$FILE
#make feltor_mpi device=omp
#export OMP_NUM_THREADS=8
#echo 2 2 7 | srun   -n 28  likwid-pin -c 0,0-7,8,8-15 ./feltor_mpi $FILE.in fq11c24t10a10z03s05.txt $SCRATCH/$FILE/$NAME.nc &> $SCRATCH/$FILE/$NAME.info



