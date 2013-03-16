
INCLUDE += 
NVCC = nvcc
NVCCFLAGS = --compiler-options -Wall -arch=sm_13
DEBUG = 

%_t: %_t.cu %.cuh
	$(NVCC) -DDG_DEBUG $< $(NVCCFLAGS) $(INCLUDE) -o $@
	./$@

%_b: %_b.cu %.cuh
	$(NVCC) -O3  $< $(NVCCFLAGS) $(INCLUDE) -o $@
	./$@

clean:
	rm -f *_t *_b
