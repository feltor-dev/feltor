device=gpu

#configure machine 
include ../../config/default.mk
include ../../config/*.mk 
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: shu_t shu_b shu2d_b shu_hpc

shu_t: shu_t.cu shu.cuh 
	$(CC) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) -DDG_DEBUG

shu_b: shu_b.cu shu.cuh 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(JSONLIB)

shu2d_b: shu2d_b.cu shu.cuh 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE)  

shu_hpc: shu_hpc.cu shu.cuh
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB)

shu_time: shu_time.cu shu.cuh
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) 
.PHONY: clean

clean:
	rm -f shu_t shu_b shu2d_b shu_hpc shu_time
