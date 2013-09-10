system=vsc # Use as Makefile parameter!
device=gpu

INCLUDE = -I../../inc # other project libraries

ifeq ($(strip $(device)),gpu)
CC = nvcc
CFLAGS = --compiler-options -Wall -arch=sm_20 -w
CFLAGS+= -Xcompiler -fopenmp 
CFLAGS+= -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP
OPT	   = -O2
else
CC = g++
CFLAGS = -Wall -fopenmp -x c++
CFLAGS+= -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
OPT	   = -O3
endif

ifeq ($(strip $(system)),home)
INCLUDE += -I/home/matthias/include # cusp and thrust libraries
#you might check the libs here, cf your glfw installation
GLFLAGS  = -lglfw -lXxf86vm -lXext -lX11 -lGLU  -lGL -lpthread -lm 
LIBS 	 = -lhdf5 -lhdf5_hl # hdf5 libraries
endif

ifeq ($(strip $(system)),leo3)
INCLUDE += -I$(HOME)/include
INCLUDE += -I$(UIBK_HDF5_INC)
INCLUDE += -I$(UIBK_OPENMPI_INC)

LIBS 	 = -L$(UIBK_HDF5_LIB) -lhdf5 -lhdf5_hl
GLFLAGS  = -lm
endif

ifeq ($(strip $(system)),vsc)
INCLUDE += -I$(HOME)/include
INCLUDE += -I/opt/sw/hdf5/1.8.11/include
INCLUDE += -I/opt/intel/impi/4.1.0/include64 
CFLAGS  += -DMPICH_IGNORE_CXX_SEEK

LIBS 	 = -L/opt/sw/hdf5/1.8.11/lib -lhdf5 -lhdf5_hl -lz
LIBS    += -L/opt/intel/composerxe/lib/intel64 -lirc -lsvml
LIBS    += -L/opt/intel/impi/4.1.0/lib64 -lmpi
GLFLAGS  = -lm
endif


all: toeflR toefl_hpc

toeflR: toeflR.cu toeflR.cuh 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) -DDG_BENCHMARK 
	echo "Attention: No compiler warnings are issued!" 

toefl_hpc: toefl_hpc.cu toeflR.cuh 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) 
	echo "Attention: No compiler warnings are issued!" 

.PHONY: clean

clean:
	rm -f toeflR toefl_hpc
