device=gpu

#configure machine
include ../../../config/default.mk
include ../../../config/*.mk
include ../../../config/devices/devices.mk

INCLUDE+= -I../../    # other project libraries

CPPFILES=$(wildcard *.cpp)
CUFILES=$(wildcard *.cu)

all: $(CPPFILES:%.cpp=%) $(CUFILES:%.cu=%)

backscatter_t: backscatter_t.cu xspacelib.h
	$(CC) $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE) -g

%_mpit: %_mpit.cu
	$(MPICC) $(OPT) $(INCLUDE) $(MPICFLAGS) $< -o $@ -g

%_mpib: %_mpib.cu
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE)

%_t: %_t.cu
	$(CC) $(OPT) $(INCLUDE) $(CFLAGS) $< -o $@ -g

%_b: %_b.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) -g

.PHONY: clean

clean:
	rm -f *_t *_b *_mpit *_mpib
