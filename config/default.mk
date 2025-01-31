# !DO NOT CHANGE THIS FILE!
# If you must, just copy one of the other *.mk files and fill in your own values
ifndef DEFAULT_MK #include guard
DEFAULT_MK=1

#compiler and compiler options
CC=g++ #C++ compiler
MPICC=mpic++  #mpi compiler
CFLAGS=-Wall -std=c++17 -mavx -mfma #flags for CC | will be added to MPICC
NVCC=nvcc #CUDA compiler
NVCCARCH=-arch sm_61 -Xcudafe "--diag_suppress=code_is_unreachable --diag_suppress=initialization_not_reachable" #nvcc gpu compute capability
NVCCFLAGS= -std=c++17 -Xcompiler "-Wall -mavx -mfma" --extended-lambda#flags for NVCC
OPT=-O2 # optimization flags for host code (it is O2 and not O3 because g++-7 up to g++-8.0 have a bug with fma in -O3, fixed in g++-8.1)
OMPFLAG=-fopenmp #openmp flag for CC and MPICC

#external libraries
INCLUDE = -I$(HOME)/include# cusp, thrust and the draw libraries
# the libhdf5-dev package installs *_serial.so libs in order to distinguish from *_openmpi.so
LIBS=-lnetcdf -lhdf5_serial -lhdf5_serial_hl # serial netcdf library for file output
LAPACKLIB=-llapacke
JSONLIB= -ljsoncpp # json library for input parameters
GLFLAGS =$$(pkg-config --static --libs glfw3) -lGL #glfw3 installation
endif # DEFAULT_MK
