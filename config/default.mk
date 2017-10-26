ifndef INCLUDED #include guard
INCLUDED=1

#default compilation values
CC=g++ #C++ compiler
CFLAGS=-Wall -x c++ -std=c++11 
MPICC=mpic++  #mpi compiler
MPICFLAGS=-Wall -x c++ -std=c++11
NVCC=nvcc #CUDA compiler
NVCCARCH=-arch sm_20 #nvcc gpu compute capability
NVCCFLAGS=-std=c++11 #
OPT=-O3 # optimization flag to be used in Makefile
OMPFLAG=-fopenmp #openmp flag for CC and MPICC

#external libraries
INCLUDE = -I$(HOME)/include# cusp, thrust, jsoncpp and the draw libraries
LIBS=-lnetcdf -lhdf5 -lhdf5_hl # netcdf library for file output
JSONLIB=-L$(HOME)/include/json/../../src/lib_json -ljsoncpp # json library for input parameters
GLFLAGS =$$(pkg-config --static --libs glfw3) #glfw3 installation
endif # INCLUDED
