
INCLUDE += 
NVCC = nvcc
NVCCFLAGS = --compiler-options -Wall -arch=sm_20 -w
DEBUG = 

version: version.cu 
	$(NVCC) $< $(NVCCFLAGS) -o $@
	./$@

cusp_thrust: cusp_thrust.cu 
	$(NVCC) $< $(NVCCFLAGS) -o $@
	./$@

dlt: dlt.cu
	$(NVCC) -O3 -w  $< $(NVCCFLAGS) $(INCLUDE) -o $@ 
	./$@

%_t: %_t.cu %.cuh
	$(NVCC) -DDG_DEBUG $< $(NVCCFLAGS) $(INCLUDE) -o $@
	./$@

%_b: %_b.cu %.cuh
	$(NVCC) -O3 -w  $< $(NVCCFLAGS) $(INCLUDE) -o $@ 
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
