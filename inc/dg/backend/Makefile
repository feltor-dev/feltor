system=home
device=omp

INCLUDE = -I../../ # other project libraries
INCLUDE+= -I../
INCLUDE+= -I./backend # header files

ifeq ($(strip $(system)),home)

INCLUDE+= -I$(HOME)/include  # cusp and thrust libraries
#you might check the libs here, cf your glfw installation
GLFLAGS   = -lglfw3 -lXxf86vm -lXext -lX11 -lGLU  -lGL -lpthread -lXrandr -lXi -lXcursor
LIBS 	 = -lnetcdf -lhdf5 -lhdf5_hl # hdf5 libraries
ifeq ($(strip $(device)),gpu)
GLFLAGS+= -lGLEW
endif
endif

ifeq ($(strip $(system)),leo3)
INCLUDE+= -I$(HOME)/include
GLFLAGS   = -lm
endif

ifeq ($(strip $(system)),vsc)
INCLUDE+= -I$(HOME)/include
GLFLAGS   = -lm
endif

ifeq ($(strip $(device)),gpu)
CC = nvcc
CFLAGS = --compiler-options -Wall -arch=sm_20 
#CFLAGS+= -Xcompiler -fopenmp 
#CFLAGS+= -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP
endif
ifeq ($(strip $(device)),omp)
CC = g++
CFLAGS = -Wall -fopenmp -x c++
CFLAGS+= -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
endif
ifeq ($(strip $(device)),mic)
CC = icc
CFLAGS = -openmp -mmic -x c++
CFLAGS+= -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
endif

CPPFILES=$(wildcard *.cpp)
CUFILES=$(wildcard *.cu)

all: $(CPPFILES:%.cpp=%) $(CUFILES:%.cu=%)

version: version.cu 
	$(CC) $(CFLAGS) $(INCLUDE) $< -o $@
	./$@

dz_2_t: dz_2_t.cu dz.cuh
	$(CC) -O2 $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE)  $(LIBS) -g 

backscatter_t: backscatter_t.cu xspacelib.cuh
	$(CC) $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE) -g

blas%_t: blas%_t.cu ../blas%.h
	$(CC) $(CFLAGS) $< -o $@ -g

%_t: %_t.cu
	$(CC) $(INCLUDE)  $(CFLAGS) $< -o $@ -g

%_b: %_b.cu 
	$(CC) -O2 $(CFLAGS) $< -o $@ $(INCLUDE) 

dg_t: dg_t.cpp dg.h 
	g++ -O3 -Wall -fopenmp  $< -o $@ $(INCLUDE) -lm

include ../mpi.mk

.PHONY: clean doc

doc: 
	doxygen Doxyfile

clean:
	rm -f *_t *_b *_mpit *_mpib cusp_thrust version
