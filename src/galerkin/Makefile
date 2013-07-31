INCLUDE = -I$(UIBK_CUSP_INC)
INCLUDE+= -I$(UIBK_HDF5_INC)
INCLUDE+= -I$(UIBK_OPENMPI_INC)
INCLUDE+= -I../../inc/# include files

LIB = -L$(UIBK_HDF5_LIB)

NVCC = nvcc
NVCCFLAGS = --compiler-options -Wall -arch=sm_20 -w

#GLFLAGS=-lglfw -lrt -lGL -lGLEW -lpthread #you might add some libs here, check your glfw installation
GLFLAGS   = -lglfw -lXxf86vm -lXext   -lX11 -lGLU   -lGL -lpthread -lm -lGLEW 

all: toeflR toefl_hpc

#toefl: toefl.cu toefl.cuh 
#	$(NVCC) -O2   $< -o $@ $(NVCCFLAGS) $(INCLUDE) $(GLFLAGS)# -DDG_BENCHMARK
#	echo "Attention: No compiler warnings are issued!" 

toeflR: toeflR.cu toeflR.cuh 
	$(NVCC) -O2  $< -o $@ $(NVCCFLAGS) $(INCLUDE) $(GLFLAGS) #-DDG_BENCHMARK 
	echo "Attention: No compiler warnings are issued!" 

toefl_hpc: toefl_hpc.cu toeflR.cuh 
	$(NVCC) -O2 $(LIB) -lhdf5 -lhdf5_hl $< -o $@ $(NVCCFLAGS) $(INCLUDE) 
	echo "Attention: No compiler warnings are issued!" 

.PHONY: clean

clean:
	rm -f toeflR toefl_hpc
