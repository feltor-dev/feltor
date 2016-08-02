
ifeq ($(strip $(SGE_CELL)),vsc2)
GLFLAGS  = -lm
CC=icc
MPICC=mpiicc 
OPT=-O3
MPICFLAGS+= -DMPICH_IGNORE_CXX_SEEK
OMPFLAG=-openmp
LIBS   = -lhdf5 -lhdf5_hl
LIBS  += -lnetcdf -lcurl
JSONLIB = -ljsoncpp
endif


#!/bin/bash
# submit with qsub 
#
# #$ -N fq18c24t10a10z03s05_FELTOR
# #$ -o fq18c24t10a10z03s05.out
# #$ -j yes
# #$ -cwd
# #$ -pe mpich 252
#
# export FILE=fq18c24t10a10z03s05
# export SCRATCH=/global/lv70607/mwiesenberge
# export NAME=n3N180N21
#
# export I_MPI_DAT_LIBRARY=/usr/lib64/libdat2.so.2
# export I_MPI_FABRICS=shm:dapl
# export I_MPI_FALLBACK=0
# export I_MPI_CPUINFO=proc
# export I_MPI_PIN_PROCESSOR_LIST=1,14,9,6,5,10,13,2,3,12,11,4,7,8,15,0
# export I_MPI_JOB_FAST_STARTUP=0
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sw/netcdf/4.3.2/lib
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/hdf5/1.8.9/intel/lib
#
#
# mkdir -p $SCRATCH/$FILE
# make feltor_mpi device=omp
# export OMP_NUM_THREADS=1
# echo 6 6 7 | mpirun -n 252 ./feltor_mpi $FILE.in fq18c24t10a10z03s05.txt $SCRATCH/$FILE/$NAME.nc &> $SCRATCH/$FILE/$NAME.info
#
