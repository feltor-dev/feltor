HOST = $(strip $(shell hostname))
ifeq ($(HOST),$(filter $(HOST),serles0 serles1))       # uniquely identify system 
INCLUDE  = -I$(HOME)/include  # locally installed libs like cusp and the draw libraries
INCLUDE += -I/usr/local/netcdf/include # parallel version of the netcdf library
INCLUDE += -I/usr/local/json/include/jsoncpp          	# cusp and thrust libraries
GLFLAGS =$$(pkg-config --static --libs glfw3) #glfw3 installation
CC=g++ #C++ compiler
MPICC=mpic++  #mpi compiler
OPT=-O3 # optimization flag
NVCCARCH=-arch sm_20 #nvcc gpu compute capability
OMPFLAG=-fopenmp #openmp flag for CC and MPICC
LIBS  = -lmpi -lcurl
LIBS += -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi -lhdf5 -lhdf5_hl  # parallel hdf5 library (debian repo)
LIBS += -L/usr/local/netcdf/lib -lnetcdf # parallel version of the netcdf library
CFLAGS+=-D_FORCE_INLINES
MPICFLAGS+=-D_FORCE_INLINES
NVCCFLAGS+=-D_FORCE_INLINES # workaround for bug in cuda 7.5 in conjunction with string.h of glibc 2.23
JSONLIB = -L/usr/local/json/lib/x86_64-linux-gnu -ljsoncpp
endif

