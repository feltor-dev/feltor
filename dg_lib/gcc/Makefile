
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
%_b: %_b.cpp %.h
	$(CXX) -O3  $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@
clean:
	rm -f *_t *_b
