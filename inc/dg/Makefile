system=vsc
device=gpu

INCLUDE = -I../../ # other project libraries
INCLUDE+= -I../
INCLUDE+= -I./backend # header files

ifeq ($(strip $(system)),home)

INCLUDE+= -I/home/matthias/include  # cusp and thrust libraries

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
CFLAGS+= -Xcompiler -fopenmp 
CFLAGS+= -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP
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
include mpi.mk

all: version cusp_thrust arakawa_t blas1_t \
	 cg2d_b cg2d_t cg_t dlt2d_b dlt2d_t dlt_t dx_t \
	 evaluation_b evaluation_t helmholtz_t backscatter_ laplace_b laplace_t \
	 operator_matrix_t operator_dynamic_b operator_t \
	 polarisation2d_b polarisation2d_t polarisation_b polarisation_t \
	 runge_kutta2d_t tensor_b toefl_b toefl_t xspacelib_t


version: version.cu 
	g++ --version
	nvcc --version
	$(CC) $(CFLAGS) $(INCLUDE) $< -o $@
	./$@

cusp_thrust: cusp_thrust.cu 
	$(CC) $(CFLAGS) $(INCLUDE) $< -o $@
	./$@

backscatter_t: backscatter_t.cu xspacelib.cuh
	$(CC) $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE)

toefl_t: toefl_t.cu toefl.cuh
	$(CC) $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE)
toefl_b: toefl_b.cu toefl.cuh
	$(CC) -O2 $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE)

blas%_t: blas%_t.cu ../blas%.h
	$(CC) $(CFLAGS) $< -o $@ 

%_t: %_t.cu
	$(CC) $(INCLUDE) -DDG_DEBUG $(CFLAGS) $< -o $@ 

#%2d_t: %2d_t.cu ../%.cuh 
#	$(CC) $(INCLUDE) -DDG_DEBUG $(CFLAGS) $< -o $@ 
#
#%3d_t: %3d_t.cu ../%.cuh 
#	$(CC) $(INCLUDE) -DDG_DEBUG $(CFLAGS) $< -o $@ 

%_b: %_b.cu 
	$(CC) -O2 $(CFLAGS) $< -o $@ $(INCLUDE) 
	echo "Attention: No compiler warnings are issued!" 

#%2d_b: %2d_b.cu ../%.h
#	$(CC) -O2 $(CFLAGS) $< -o $@ $(INCLUDE)
#	echo "Attention: No compiler warnings are issued!" 
#
#%3d_b: %3d_b.cu ../%.h
#	$(CC) -O2 $(CFLAGS) $< -o $@ $(INCLUDE)
#	echo "Attention: No compiler warnings are issued!" 

dg_t: dg_t.cpp ../dg.h 
	g++ -O3 -Wall -fopenmp  $< -o $@ $(INCLUDE) -lm
	#g++ -O3 $(CFLAGS) $< -o $@ $(INCLUDE) -lm

bath_t: bathRZ_t.cu 
	$(CC) -O2 $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE) 
	echo "Attention: No compiler warnings are issued!" 
	
geometry_t: geometry_t.cu 
	$(CC) -O2 $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE) 
	echo "Attention: No compiler warnings are issued!" 

.PHONY: clean doc

dz_geom_t: dz_geom_t.cu 
	$(CC) -O2 $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE) 
	echo "Attention: No compiler warnings are issued!" 

doc: 
	doxygen Doxyfile

clean:
	rm -f *_t *_b *_mpit *_mpib cusp_thrust version
