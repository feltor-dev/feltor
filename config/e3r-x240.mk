ifeq ($(strip $(shell hostname)),e3r-x240)   # uniquely identify system
CC=g++                                       # the host c++ compiler
OPT=-O3                                      # the optimization flag for the host
OMPFLAG=-fopenmp                             # openmp flag for CC and MPICC
LIBS=-lnetcdf -lhdf5 -lhdf5_hl               # netcdf library for file output
GLFLAGS = -lglfw -lGL
endif
