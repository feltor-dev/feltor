device=gpu

#configure machine 
include ../../config/default.mk
include ../../config/*.mk 
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: shu_t shu_b shu2d_b shu_hpc shu_dat

shu_t: shu_t.cu shu.cuh 
	$(CC) $< -o $@ $(CFLAGS) $(INCLUDE) $(GLFLAGS) -DDG_DEBUG

shu_b: shu_b.cu shu.cuh 
	$(CC) $(OPT) $< -o $@ $(CFLAGS) $(INCLUDE) $(GLFLAGS) $(JSONLIB)

shu2d_b: shu2d_b.cu shu.cuh 
	$(CC) $(OPT) $< -o $@ $(CFLAGS) $(INCLUDE)  

shu_hpc: shu_hpc.cu shu.cuh
	$(CC) $(OPT) $< -o $@ $(CFLAGS) $(INCLUDE) $(LIBS) $(JSONLIB)

shu_dat: shu_dat.cu 
	$(CC) $(OPT) $< -o $@ $(CFLAGS) $(INCLUDE) $(LIBS)

shu_time: shu_time.cu shu.cuh
	$(CC) $(OPT) $< -o $@ $(CFLAGS) $(INCLUDE) 
.PHONY: clean

clean:
	rm -f shu_t shu_b shu2d_b shu_hpc shu_dat shu_time
