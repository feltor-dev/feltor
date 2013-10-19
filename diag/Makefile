INCLUDE = -I../inc/# include files libs
INCLUDE+= -I../src/# include files from source code
INCLUDE+= -I/home/matthias/include

CXX = g++
NVCC = nvcc
NVCCFLAGS = --compiler-options -Wall -arch=sm_20 -w

#GLFLAGS=-lglfw -lrt -lGL -lGLEW -lpthread #you might add some libs here, check your glfw installation
GLFLAGS   = -lglfw -lXxf86vm -lXext   -lX11 -lGLU   -lGL -lpthread -lm -lGLEW 

all: toefl_video innto_video show com


toefl_video: toefl_video.cu 
	$(NVCC) -O2 -lhdf5 -lhdf5_hl $< -o $@ $(NVCCFLAGS) $(INCLUDE) $(GLFLAGS)
	echo "Attention: No compiler warnings are issued!" 

toefl_analytics: toefl_analytics.cu 
	$(NVCC) -O2 -lhdf5 -lhdf5_hl $< -o $@ $(NVCCFLAGS) $(INCLUDE) $(GLFLAGS)
toefl_compare: toefl_compare.cu 
	$(NVCC) -O2 -lhdf5 -lhdf5_hl $< -o $@ $(NVCCFLAGS) $(INCLUDE) $(GLFLAGS)

innto_video: innto_video.cpp
	$(CXX) -O3  $< -o $@ -std=c++0x $(INCLUDE) $(GLFLAGS) -lhdf5 -lhdf5_hl -lfftw3
	echo "Attention: No compiler warnings are issued!" 

show: show.cpp
	$(CXX) $< -o $@ -std=c++0x $(INCLUDE) -lhdf5 -lhdf5_hl -Wall

com: com.cu
	$(NVCC) -O2 -lhdf5 -lhdf5_hl $< -o $@ $(NVCCFLAGS) $(INCLUDE) 
	echo "Attention: No compiler warnings are issued!" 


.PHONY: clean

clean:
	rm -f toefl_video innto_video show 
