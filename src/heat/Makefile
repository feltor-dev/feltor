device=gpu

#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: heat heat_hpc heat_hpc

heat: heat.cu heat.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(JSONLIB) $(GLFLAGS) -g

heat_hpc: heat_hpc.cpp heat.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(JSONLIB) $(LIBS)

.PHONY: clean

clean:
	rm -f heat heat_hpc
