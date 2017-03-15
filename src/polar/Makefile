device=omp
OS=ubuntu1504

#configure machine 
include ../../config/default.mk
include ../../config/*.mk 
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries


all: polar

polar: polar.cu ns.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@  $(INCLUDE) $(GLFLAGS) $(JSONLIB)
.PHONY: clean

clean:
	rm -f polar
