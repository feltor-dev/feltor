
ifeq ($(strip $(shell domainname)),leo3-domain)
INCLUDE  = -I$(HOME)/include
INCLUDE += -I$(UIBK_HDF5_INC)
INCLUDE += -I$(UIBK_OPENMPI_INC)
INCLUDE += -I$(UIBK_NETCDF_4_INC)
GLFLAGS  = -lm
CC=g++
MPICC=mpicxx
OPT=-O3
NVCCARCH=-arch=sm_20 #Tesla M2090
OMPFLAG=-fopenmp
LIBS 	 = -L$(UIBK_HDF5_LIB) -lhdf5 -lhdf5_hl 
LIBS 	+= -L$(UIBK_NETCDF_4_LIB) -lnetcdf -lcurl -lm
endif
##########################modules to load#############################
#module load netcdf-4/4.3.2
#module load cuda/5.0
#######################Submit script for mpi+gpu job###########################
##!/bin/bash
#
##$ -N gpu_benchmark
##$ -o benchmark_gpu.txt
##$ -l h_rt=00:40:00
##$ -j yes
##$ -cwd
#
##$ -l gpu=2
#
##reserve nodes 
##$ -pe openmpi-2perhost 2
##$ -l h_vmem=5G
#
#export FILE=benchmark_gpu.dat
#
#make cluster_mpib device=gpu
#mv cluster_mpib cluster_gpub
#echo "# npx npy npz #procs n Nx Ny Nz t_AXPBY t_DOT t_DX t_DY t_DZ t_ARAKAWA #iterations t_1xELLIPTIC_CG t_DS" > $FILE
#echo 1 1 2  3 102 102 18 | mpirun -n 2 ./cluster_gpub >> $FILE
