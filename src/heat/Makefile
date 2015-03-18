system=home
device=gpu

INCLUDE = -I../../inc # other project libraries
INCLUDE+= -I../ # geometry library


ifeq ($(strip $(device)),gpu)
CC = nvcc
CFLAGS = --compiler-options -Wall -arch=sm_20 -w
CFLAGS+= -Xcompiler -fopenmp 
CFLAGS+= -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP
GLFLAGS= -lGLEW
OPT	   = -O2
endif
ifeq ($(strip $(device)),omp)
CC = g++
CFLAGS = -Wall -fopenmp -x c++
CFLAGS+= -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
OPT	   = -O3
endif
ifeq ($(strip $(device)),mpi)
CC = mpic++
CFLAGS = #-Wall -x c++
CFLAGS+= -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
OPT	   = -O3 
endif

ifeq ($(strip $(system)),home)
INCLUDE += -I$(HOME)/include # cusp and thrust libraries

#you might check the libs here, cf your glfw installation
GLFLAGS += -lglfw3 -lXxf86vm -lXext -lX11 -lGLU  -lGL -lpthread -lXi -lXrandr -lXcursor
LIBS 	 = -lnetcdf #-lhdf5 -lhdf5_hl # hdf5 libraries
endif

ifeq ($(strip $(system)),leo3)
INCLUDE += -I$(HOME)/include
INCLUDE += -I$(UIBK_HDF5_INC)
INCLUDE += -I$(UIBK_OPENMPI_INC)
INCLUDE += -I$(UIBK_NETCDF_4_INC)

LIBS 	 = -L$(UIBK_HDF5_LIB) -lhdf5 -lhdf5_hl 
LIBS 	+= -L$(UIBK_NETCDF_4_LIB) -lnetcdf -lcurl -lm
GLFLAGS  = -lm
endif

ifeq ($(strip $(system)),vsc)
INCLUDE += -I$(HOME)/include
INCLUDE += -I/opt/sw/hdf5/1.8.11/include
INCLUDE += -I/opt/intel/impi/4.1.0/include64 
INCLUDE += -I/opt/sw/netcdf/4.0.1/intel-hdf5/include
CFLAGS  += -DMPICH_IGNORE_CXX_SEEK

LIBS 	 = -L/opt/sw/hdf5/1.8.11/lib -lhdf5 -lhdf5_hl -lz
LIBS    += -L/opt/intel/composerxe/lib/intel64 -lirc -lsvml
LIBS    += -L/opt/intel/impi/4.1.0/lib64 -lmpi
LIBS    += -L/opt/sw/netcdf/4.0.1/intel-hdf5/lib -lnetcdf
GLFLAGS  = -lm
endif

all: heat

heat: heat.cu heat.cuh 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) -DDG_BENCHMARK 
	echo "Attention: No compiler warnings are issued!" 
	
heat_hpc: heat_hpc.cu heat.cuh 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_BENCHMARK
	echo "Attention: No compiler warnings are issued!" 
	
.PHONY: clean
	
clean:
	rm -f heat heat_hpc geometry_t dz_geom_t
