device=gpu

#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: shu shu_hpc

shu: shu.cpp shu.h init.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(GLFLAGS) $(JSONLIB) -g

shu_hpc: shu.cpp shu.h init.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g -DWITHOUT_GLFW

.PHONY: clean

clean:
	rm -f shu shu_hpc
