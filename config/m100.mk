ifeq ($(strip $(HPC_SYSTEM)),m100)
#CC=xlc++ #C++ compiler
#MPICC=mpixlC  #mpi compiler
#CFLAGS=-Wall -std=c++1y -DWITHOUT_VCL -mcpu=power9 -qstrict# -mavx -mfma #flags for CC
#OMPFLAG=-qsmp=omp
#OPT=-O3 # optimization flags for host code
#NVCC=nvcc #CUDA compiler
#NVCCARCH=-arch sm_70 -Xcudafe "--diag_suppress=code_is_unreachable --diag_suppress=initialization_not_reachable" #nvcc gpu compute capability
#NVCCFLAGS= -std=c++14 -Xcompiler "-mcpu=power9 -Wall"# -mavx -mfma" #flags for NVCC
CFLAGS=-Wall -std=c++14 -DWITHOUT_VCL -mcpu=power9 # -mavx -mfma #flags for CC
OPT=-O3 # optimization flags for host code
NVCC=nvcc #CUDA compiler
NVCCARCH=-arch sm_70 -Xcudafe "--diag_suppress=code_is_unreachable --diag_suppress=initialization_not_reachable" #nvcc gpu compute capability
NVCCFLAGS= -std=c++14 -Xcompiler "-mcpu=power9 -Wall"# -mavx -mfma" #flags for NVCC

INCLUDE += -I$(NETCDF_INC) -I$(HDF5_INC)
JSONLIB=-L$(HOME)/include/json/../../lib -ljsoncpp_static # json library for input parameters
LIBS    +=-L$(HDF5_LIB) -lhdf5 -lhdf5_hl
LIBS    +=-L$(NETCDF_LIB) -lnetcdf -lcurl
#is the novel jsoncpp lib folder changed?
endif
