INCLUDE = -I../inc/# include files libs
INCLUDE+= -I../src/# include files from source code

NVCC = nvcc
NVCCFLAGS = --compiler-options -Wall -arch=sm_20 -w

#GLFLAGS=-lglfw -lrt -lGL -lGLEW -lpthread #you might add some libs here, check your glfw installation
GLFLAGS   = -lglfw -lXxf86vm -lXext   -lX11 -lGLU   -lGL -lpthread -lm -lGLEW 

toefl_video: toefl_video.cu 
	$(NVCC) -O2 -lhdf5 -lhdf5_hl $< -o $@ $(NVCCFLAGS) $(INCLUDE) $(GLFLAGS)
	echo "Attention: No compiler warnings are issued!" 

.PHONY: clean

clean:
	rm -f toefl_video 
