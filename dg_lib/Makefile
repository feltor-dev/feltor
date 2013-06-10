INCLUDE = -I/home/matthias/include # Eigen library
INCLUDE+= -I../# draw library

INCLUDE += -Invcc
NVCC = nvcc
NVCCFLAGS = --compiler-options -Wall -arch=sm_20 -w

#GLFLAGS=-lglfw -lrt -lGL -lGLEW -lpthread #you might add some libs here, check your glfw installation
GLFLAGS   = -lglfw -lXxf86vm -lXext   -lX11 -lGLU   -lGL -lpthread -lm -lGLEW 

toefl: toefl.cu toefl.cuh 
	$(NVCC) -O3   $< -o $@ $(NVCCFLAGS) $(INCLUDE) $(GLFLAGS)
	echo "Attention: No compiler warnings are issued!" 

toeflR: toeflR.cu toeflR.cuh 
	$(NVCC) -O2  $< -o $@ $(NVCCFLAGS) $(INCLUDE) $(GLFLAGS)
	echo "Attention: No compiler warnings are issued!" 

toefl_hpc: toefl_hpc.cu toefl.cuh 
	$(NVCC) -O2 -lhdf5 -lhdf5_hl $< -o $@ $(NVCCFLAGS) $(INCLUDE) 
	echo "Attention: No compiler warnings are issued!" 
.PHONY: clean doc

doc: 
	doxygen Doxyfile

clean:
	rm -f toefl
