ifeq ($(strip $(HPC_SYSTEM)),davide)
CFLAGS=-Wall -std=c++14 -DWITHOUT_VCL -mcpu=power8 # -mavx -mfma #flags for CC
OPT=-O3 # optimization flags for host code
NVCC=nvcc #CUDA compiler
NVCCARCH=-arch sm_60 -Xcudafe "--diag_suppress=code_is_unreachable --diag_suppress=initialization_not_reachable" #nvcc gpu compute capability
NVCCFLAGS= -std=c++14 -Xcompiler "-mcpu=power8 -Wall"# -mavx -mfma" #flags for NVCC
INCLUDE += -I$(NETCDF_INC) -I$(HDF5_INC)
LIBS    +=-L$(HDF5_LIB) -lhdf5 -lhdf5_hl
LIBS    +=-L$(NETCDF_LIB) -lnetcdf -lcurl
endif
