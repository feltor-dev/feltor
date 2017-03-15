ifeq ($(OS),arch)
#default machine values
INCLUDE = -I$(HOME)/include#  # cusp and thrust and the draw libraries
INCLUDE += -I/usr/include/hdf5/serial -I/opt/cuda/include/
GLFLAGS =$$(pkg-config --static --libs glfw3) -lGL #glfw3 installation
CFLAGS += -x c++
CC=g++ #C++ compiler
MPICC=mpic++  #mpi compiler
OPT=-O3 # optimization flag
NVCCARCH=-arch sm_20 #nvcc gpu compute capability
OMPFLAG=-fopenmp #openmp flag for CC and MPICC
LIBS=-lnetcdf -lhdf5_serial -lhdf5_serial_hl # netcdf library for file output
JSONLIB=-I/usr/include/jsoncpp -ljsoncpp # json library for input parameters
endif # ubuntu1504
