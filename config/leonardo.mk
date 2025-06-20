ifeq ($(strip $(HPC_SYSTEM)),leonardo)
CFLAGS=-Wall -std=c++17 -mavx -mfma #flags for CC
OPT=-O3 # optimization flags for host code
NVCC=nvcc #CUDA compiler
NVCCARCH=-arch sm_80 -Xcudafe "--diag_suppress=code_is_unreachable --diag_suppress=initialization_not_reachable" #nvcc gpu compute capability
NVCCFLAGS= -std=c++17 -Xcompiler "-Wall -mavx -mfma" --extended-lambda# -mavx -mfma" #flags for NVCC

INCLUDE += -I$(NETCDF_C_INC) -I$(HDF5_INC)
INCLUDE += -I$(BOOST_INC) #-I$(LAPACK_INC)
JSONLIB= -ljsoncpp # json library for input parameters
LIBS    =-L$(HDF5_LIB) -lhdf5 -lhdf5_hl
LIBS    +=-L$(NETCDF_C_LIB) -lnetcdf -lcurl
#LAPACKLIB = -L$(LAPACK_LIB) -llapack
endif
#########################Modules to load ##################
# module load gcc/12.2
# module load cuda/12.1
# 
# module load spack
# # spack list jsoncpp
# # spack spec -Il jsoncpp
# # spack install jsoncpp %gcc@12.2.0
# spack load jsoncpp%gcc@12.2.0
# module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8
# module load boost/1.80.0--gcc--11.3.0
# module load hdf5/1.12.2--gcc--11.3.0
# module load netcdf-c/4.9.0--gcc--11.3.0


