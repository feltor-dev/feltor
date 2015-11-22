


ifeq ($(strip $(shell hostname)),c722-pc161)#    # uniquely identify system
INCLUDE = -I$(HOME)/include#                    # cusp and thrust libraries
GLFLAGS =$$(pkg-config --static --libs glfw3)#  # flags for glfw3
CC=g++#                                         # the host c++ compiler
MPICC=mpic++#                                   # the mpi compiler
OPT=-O3#                                        # the optimization flag for the host
NVCCARCH=-arch=sm_35#                           # nvcc gpu compute capability
OMPFLAG=-fopenmp#                               # openmp flag for CC and MPICC
LIBS=-lnetcdf -lhdf5 -lhdf5_hl#                 # netcdf library for file output
endif
