device=omp

#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: shu_b

shu_b: shu_b.cu shu.cuh init.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(GLFLAGS) $(JSONLIB) -g

shu_hpc: shu_b.cu shu.cuh init.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g -DDG_BENCHMARK -DWITHOUT_GLFW

.PHONY: clean

clean:
	rm -f shu_b
