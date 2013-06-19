INCLUDE = -I/home/matthias/include # Eigen library
INCLUDE += -I../ # other libraries
NVCC = nvcc
NVCCFLAGS = --compiler-options -Wall -arch=sm_20

CXX = g++
CFLAGS = -Wall -fopenmp
CINCLUDE = -I/usr/local/cuda/include/

#GLFLAGS=-lglfw -lrt -lGL -lGLEW -lpthread #you might add some libs here, check your glfw installation
GLFLAGS   = -lglfw -lXxf86vm -lXext -lX11 -lGLU  -lGL -lpthread -lm 
LIBS = cusp_eigen.o

version: version.cu 
	$(NVCC) $< $(NVCCFLAGS) -o $@
	./$@

cusp_thrust: cusp_thrust.cu 
	$(NVCC) $< $(NVCCFLAGS) -o $@
	./$@

host_window_t: host_window_t.cu
	$(NVCC) $< $(NVCCFLAGS) -o $@ $(GLFLAGS) $(INCLUDE)


blas%_t: blas%_t.cu blas%.h
	$(NVCC) $< $(NVCCFLAGS) -o $@ 

%.o: %.cpp
	$(CXX) -O3 -c $< $(CFLAGS) $(CINCLUDE) $(INCLUDE) 

%_t: %_t.cu %.cuh $(LIBS)
	$(NVCC) -DDG_DEBUG $< $(LIBS) -o $@ $(NVCCFLAGS) $(GLFLAGS)  $(INCLUDE) 
	./$@

%_b: %_b.cu %.cuh $(LIBS)
	$(NVCC) -O2 -w  $< -o $@ $(NVCCFLAGS) $(INCLUDE) $(LIBS) $(GLFLAGS)
	echo "Attention: No compiler warnings are issued!" 
	./$@

%2d_t: %2d_t.cu %.cuh $(LIBS)
	$(NVCC) -DDG_DEBUG $< -o $@ $(NVCCFLAGS) $(LIBS) $(INCLUDE)
	./$@

%2d_b: %2d_b.cu %.cuh $(LIBS)
	$(NVCC) -O2 -w  $< -o $@ $(NVCCFLAGS) $(LIBS) $(INCLUDE)
	echo "Attention: No compiler warnings are issued!" 
	./$@

.PHONY: clean doc

doc: 
	doxygen Doxyfile

clean:
	rm -f *_t *_b cusp_thrust version
