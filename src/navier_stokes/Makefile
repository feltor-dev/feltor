device=gpu

#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: navier_stokes navier_stokes_hpc

navier_stokes: ../feltor/feltor.cu ../feltor/init.h navier_stokes.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(LIBS) $(JSONLIB) -DDG_BENCHMARK -g -DWITH_NAVIER_STOKES

navier_stokes_hpc: ../feltor/feltor.cu ../feltor/init.h navier_stokes.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DDG_BENCHMARK -DWITHOUT_GLFW -DWITH_NAVIER_STOKES

.PHONY: clean

clean:
	rm -f navier_stokes navier_stokes_hpc
