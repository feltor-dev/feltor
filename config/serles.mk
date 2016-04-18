HOST = $(strip $(shell hostname))
ifeq ($(HOST),$(filter $(HOST),serles0 serles1))       # uniquely identify system 
INCLUDE  = -I$(HOME)/include
INCLUDE += -I/usr/include/hdf5/serial          	# cusp and thrust libraries
GLFLAGS =$$(pkg-config --static --libs glfw3)	# flags for glfw3
CC=gcc#                                         # the host c++ compiler
CFLAGS = -D_FORCE_INLINES			#
MPICC=mpic++#                                   # the mpi compiler
OPT=-O3#                                        # the optimization flag for the host
NVCCARCH=-arch=sm_20#                           # nvcc gpu compute capability
OMPFLAG=-fopenmp#                               # openmp flag for CC and MPICC
LIBS=-lnetcdf -L/usr/include/hdf5/serial -lhdf5 -lhdf5_hl#                 # netcdf library for file output
endif
