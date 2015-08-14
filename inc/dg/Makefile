system=home
device=gpu


INCLUDE = -I../../ # other project libraries
INCLUDE = -I../ # other project libraries

ifeq ($(strip $(system)),home)

INCLUDE+= -I$(HOME)/include  # cusp and thrust libraries

#you might check the libs here, cf your glfw installation
GLFLAGS   = -lglfw3 -lXxf86vm -lXext -lX11 -lGLU  -lGL -lpthread -lXrandr -lXi
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
#CFLAGS+= -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP
#CFLAGS+= -DCUSP_DEVICE_BLAS_SYSTEM=CUSP_DEVICE_BLAS_CUBLAS -lcublas
#CFLAGS+= -DCUSP_USE_TEXTURE_MEMORY
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

include mpi.mk

toefl_t: toefl_t.cu toefl.cuh
	$(CC) $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE)
toefl_b: toefl_b.cu toefl.cuh
	$(CC) -O2 $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE) 

%_t: %_t.cu
	$(CC) -O2 $(INCLUDE) -DDG_DEBUG $(CFLAGS) $< -o $@ 

%_b: %_b.cu 
	$(CC) -O2 $(CFLAGS) $< -o $@ $(INCLUDE) 

dg_t: dg_t.cpp ../dg.h 
	g++ -O3 -Wall -fopenmp  $< -o $@ $(INCLUDE) -lm

bathRZ_t: bathRZ_t.cu 
	$(CC) -O2 $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE) 
	
.PHONY: clean doc

dz_geom_t: dz_geom_t.cu 
	$(CC) -O2 $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE) 

doc: 
	doxygen Doxyfile

clean:
	rm -f *_t *_b *_mpit *_mpib 
