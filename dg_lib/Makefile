INCLUDE = -I/home/matthias/include # Eigen library
INCLUDE += -Invcc
NVCC = nvcc
NVCCFLAGS = --compiler-options -Wall -arch=sm_20 -w

#GLFLAGS=-lglfw -lrt -lGL -lGLEW -lpthread #you might add some libs here, check your glfw installation
GLFLAGS   = -lglfw -lXxf86vm -lXext   -lX11 -lGLU   -lGL -lpthread -lm -lGLEW 

toefl: toefl.cu toefl.cuh 
	$(NVCC) -O3   $< -o $@ $(NVCCFLAGS) $(INCLUDE) $(GLFLAGS)
	echo "Attention: No compiler warnings are issued!" 

.PHONY: clean doc

doc: 
	doxygen Doxyfile

clean:
	rm -f toefl
