INCLUDE = -I/home/matthias/include # Eigen library
NVCC = nvcc
NVCCFLAGS = --compiler-options -Wall -arch=sm_20 -w

CXX = g++
CFLAGS = -Wall -fopenmp
CINCLUDE = -I/usr/local/cuda/include/

#GLFLAGS=-lglfw -lrt -lGL -lGLEW -lpthread #you might add some libs here, check your glfw installation
GLFLAGS   = -lglfw -lXxf86vm -lXext   -lX11 -lGLU   -lGL -lpthread -lm -lGLEW 

version: version.cu 
	$(NVCC) $< $(NVCCFLAGS) -o $@
	./$@

cusp_thrust: cusp_thrust.cu 
	$(NVCC) $< $(NVCCFLAGS) -o $@
	./$@

%_eigen_t: %_eigen_t.cpp %_eigen.h 
	$(CXX) -DDG_DEBUG $< $(CFLAGS) $(CINCLUDE) $(INCLUDE) -o $@
	./$@

%_t: %_t.cu %.cuh
	$(NVCC) -DDG_DEBUG $< $(NVCCFLAGS) $(GLFLAGS) $(INCLUDE) -o $@
	./$@

%_b: %_b.cu %.cuh
	$(NVCC) -O3 -w  $< $(NVCCFLAGS) $(INCLUDE) $(GLFLAGS)-o $@ 
	echo "Attention: No compiler warnings are issued!" 
	./$@

%2d_t: %2d_t.cu %.cuh
	$(NVCC) -DDG_DEBUG $< $(NVCCFLAGS) $(INCLUDE) -o $@
	./$@

%2d_b: %2d_b.cu %.cuh
	$(NVCC) -O3 -w  $< $(NVCCFLAGS) $(INCLUDE) -o $@ 
	echo "Attention: No compiler warnings are issued!" 
	./$@

.PHONY: clean doc

doc: 
	doxygen Doxyfile

clean:
	rm -f *_t *_b cusp_thrust version
