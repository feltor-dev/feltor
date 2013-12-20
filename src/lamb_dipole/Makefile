INCLUDE = -I../../inc/# include files
INCLUDE += -I/$(HOME)/include

NVCC = nvcc
NVCCFLAGS = --compiler-options -Wall -arch=sm_20 

#GLFLAGS=-lglfw -lrt -lGL -lGLEW -lpthread #you might add some libs here, check your glfw installation
GLFLAGS   = -lglfw -lXxf86vm -lXext   -lX11 -lGLU   -lGL -lpthread -lm -lGLEW 

all: shu_t shu_b shu2d_b shu_hpc shu_dat

shu_t: shu_t.cu shu.cuh 
	$(NVCC) $< -o $@ $(NVCCFLAGS) $(INCLUDE) $(GLFLAGS) -DDG_DEBUG

shu_b: shu_b.cu shu.cuh 
	$(NVCC) -O2 $< -o $@ $(NVCCFLAGS) -w $(INCLUDE) $(GLFLAGS)
	echo "Attention: No compiler warnings are issued!" 

shu2d_b: shu2d_b.cu shu.cuh 
	$(NVCC) -O2 $< -o $@ $(NVCCFLAGS) -w $(INCLUDE) 
	echo "Attention: No compiler warnings are issued!" 

shu_hpc: shu_hpc.cu shu.cuh
	$(NVCC) -O2 $< -o $@ $(NVCCFLAGS) -w $(INCLUDE) -lhdf5 -lhdf5_hl

shu_dat: shu_dat.cu shu.cuh
	$(NVCC) -O2 $< -o $@ $(NVCCFLAGS) -w $(INCLUDE) -lhdf5 -lhdf5_hl

shu_time: shu_time.cu shu.cuh
	$(NVCC) -O2 $< -o $@ $(NVCCFLAGS) -w $(INCLUDE) 
.PHONY: clean

clean:
	rm -f shu_t shu_b shu2d_b shu_hpc shu_dat shu_time
