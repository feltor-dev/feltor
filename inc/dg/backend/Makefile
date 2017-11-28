device=gpu

#configure machine 
include ../../../config/default.mk
include ../../../config/*.mk 
include ../../../config/devices/devices.mk

INCLUDE+= -I../../ # other project libraries
INCLUDE+= -I../    # other project libraries

CPPFILES=$(wildcard *.cpp)
CUFILES=$(wildcard *.cu)

all: $(CPPFILES:%.cpp=%) $(CUFILES:%.cu=%)

version: version.cu 
	$(CC) $(CFLAGS) $(INCLUDE) $< -o $@
	./$@

backscatter_t: backscatter_t.cu xspacelib.cuh
	$(CC) $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE) -g

blas%_t: blas%_t.cu ../blas%.h
	$(CC) $(CFLAGS) $< -o $@ -g

%_t: %_t.cu
	$(CC) $(INCLUDE)  $(CFLAGS) $< -o $@ -g 
	
%_mpit: %_mpit.cu
	$(MPICC) -O2 $(INCLUDE) -DDG_DEBUG $(MPICFLAGS) $< -o $@ -g

%_mpib: %_mpib.cu
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) 
	
%_b: %_b.cu 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) 

.PHONY: clean doc

doc: 
	doxygen Doxyfile

clean:
	rm -f *_t *_b *_mpit *_mpib cusp_thrust version
