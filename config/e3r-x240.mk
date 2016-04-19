ifeq ($(strip $(shell hostname)),e3r-x240)   # uniquely identify system
CC=g++                                       # the host c++ compiler
MPICC = mpic++
MPICFLAGS+= -DMPICH_IGNORE_CXX_SEEK
OPT=-O3                                      # the optimization flag for the host
OMPFLAG=-fopenmp                             # openmp flag for CC and MPICC
LIBS=-lnetcdf -lcurl -lhdf5 -lhdf5_hl               # netcdf library for file output
GLFLAGS = -lglfw -lGL
endif
