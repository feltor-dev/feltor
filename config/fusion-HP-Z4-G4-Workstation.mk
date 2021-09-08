ifeq ($(strip $(shell hostname)),fusion-HP-Z4-G4-Workstation)#    # uniquely identify system

#compiler and compiler options
CC=g++ #C++ compiler
MPICC=mpic++  #mpi compiler
CFLAGS=-Wall -std=c++14 -mavx -mfma #flags for CC
NVCC=nvcc #CUDA compiler
NVCCARCH=-arch sm_61 -Xcudafe "--diag_suppress=code_is_unreachable --diag_suppress=initialization_not_reachable" #nvcc gpu compute capability
NVCCFLAGS= -std=c++14 -Xcompiler "-Wall -mavx -mfma" #flags for NVCC
OPT=-O2 # optimization flags for host code (it is O2 and not O3 because g++-7 up to g++-8.0 have a bug with fma in -O3, fixed in g++-8.1)
OMPFLAG=-fopenmp #openmp flag for CC and MPICC

#external libraries
INCLUDE = -I$(HOME)/include# cusp, thrust, jsoncpp and the draw libraries
LIBS=-lnetcdf -lhdf5 -lhdf5_hl -L /usr/lib/x86_64-linux-gnu/hdf5/serial  # netcdf library for file output
JSONLIB=-L$(HOME)/include/json/../../src/lib_json -ljsoncpp # json library for input parameters
GLFLAGS =$$(pkg-config --static --libs glfw3) -lGL #glfw3 installation

endif
