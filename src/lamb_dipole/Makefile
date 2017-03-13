device=omp
OS=ubuntu1504

#configure machine 
include ../../config/default.mk
include ../../config/*.mk 
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries


all: shu_t shu_b shu2d_b shu_hpc shu_dat

shu_t: shu_t.cu shu.cuh 
	$(CC) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) -DDG_DEBUG

shu_b: shu_b.cu shu.cuh 
	$(CC) $(OPT) $(CFLAGS) $< -o $@  $(INCLUDE) $(GLFLAGS)

shu2d_b: shu2d_b.cu shu.cuh 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE)

shu_hpc: shu_hpc.cu shu.cuh
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS)

shu_dat: shu_dat.cu 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS)

shu_time: shu_time.cu shu.cuh
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE)
.PHONY: clean

clean:
	rm -f shu_t shu_b shu2d_b shu_hpc shu_dat shu_time
