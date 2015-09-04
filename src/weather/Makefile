device=gpu

#configure machine 
include ../../config/default.mk
include ../../config/*.mk 
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: convection

convection: convection.cu convection.cuh 
	$(CC)  $(OPT) -DDG_BENCHMARK $(CFLAGS) $(INCLUDE)$< -o $@ $(GLFLAGS)   

.PHONY: clean

clean:
	rm -f convection
