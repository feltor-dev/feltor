device=gpu

#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: shu_b

shu_b: shu_b.cu shu.cuh
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(GLFLAGS) $(JSONLIB) -g -DDG_BENCHMARK

.PHONY: clean

clean:
	rm -f shu_b
