system=home # Use as Makefile parameter!
device=gpu

INCLUDE = -I../../inc/ # other project libraries

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
INCLUDE += -I$(HOME)/include # cusp and thrust libraries
#you might check the libs here, cf your glfw installation
GLFLAGS  = $$(pkg-config --static --libs glfw3)
LIBS 	 = -lhdf5 -lhdf5_hl # hdf5 libraries
endif

ifeq ($(strip $(system)),leo3)
INCLUDE += -I$(HOME)/include
INCLUDE += -I$(UIBK_HDF5_INC)
INCLUDE += -I$(UIBK_OPENMPI_INC)

LIBS 	+= -L$(UIBK_HDF5_LIB) -lhdf5 -lhdf5_hl
GLFLAGS  = -lm
endif

ifeq ($(strip $(system)),vsc)
INCLUDE += -I$(HOME)/include

GLFLAGS  = -lm
endif


all: convection

convection: convection.cu convection.cuh 
	$(CC)  $(OPT) -DDG_BENCHMARK $(CFLAGS) $(INCLUDE)$< -o $@ $(GLFLAGS)   
	echo "Attention: No compiler warnings are issued!" 

.PHONY: clean

clean:
	rm -f convection
