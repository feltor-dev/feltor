
INCLUDE += 
CXX = g++ 
NVCC = nvcc
CFLAGS = -Wall -std=c++0x
NVCCFLAGS = --compiler-options -Wall -arch=sm_13
LIBS = -lfftw3 -lm
DEBUG = 

%_t: %_t.cpp %.h
	$(CXX) -DDG_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@

quadmat_t: quadmat_t.cu quadmat.cuh
	$(NVCC) $< $(NVCCFLAGS) $(INCLUDE) -o $@
	./$@
quadmat_b: quadmat_b.cu quadmat.cuh
	$(NVCC) -O3 $< $(NVCCFLAGS) $(INCLUDE) -o $@
	./$@
array_t: array_t.cu array.h
	$(NVCC) -DDG_DEBUG $< $(NVCCFLAGS) $(INCLUDE) -o $@
	./$@
array_b: array_b.cu array.h
	$(NVCC) -O3 $< $(NVCCFLAGS) $(INCLUDE) -o $@
	./$@

%_b: %_b.cpp %.h
	$(CXX) -O3  $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@
clean:
	rm -f *_t *_b
