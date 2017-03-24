ifeq ($(strip $(shell hostname)),tek-ift-ans89)#    # uniquely identify system
INCLUDE = -I$(HOME)/include#                    # cusp and thrust libraries
INCLUDE += -I/home/rku000/local/src/jsoncpp/include
#INCLUDE += -I/home/rku000/local/include
INCLUDE += -I/home/rku000/local
INCLUDE += -I/home/rku000/local/cusplibrary
INCLUDE += -I/home/rku000/local/thrust
GLFLAGS =$$(pkg-config --static --libs glfw3)#  # flags for glfw3
CC=clang-3.8#                                         # the host c++ compiler
MPICC=mpic++#                                   # the mpi compiler
OPT=-O3#                                        # the optimization flag for the host
NVCCARCH=-arch=sm_50#                           # nvcc gpu compute capability
OMPFLAG=-fopenmp=libomp#                               # openmp flag for CC and MPICC
LIBS=-lnetcdf -lhdf5 -lhdf5_hl#                 # netcdf library for file output
JSONLIB=-L/home/rku000/local/src/jsoncpp/src/lib_json -ljsoncpp # json library for input parameters
endif
