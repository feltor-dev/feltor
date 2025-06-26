device=gpu

#configure machine
include ../../config/default.mk
include ../../config/version.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: navier_stokes navier_stokes_hpc

navier_stokes: ../feltor/feltor.cpp ../feltor/init.h navier_stokes.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(LIBS) $(JSONLIB) -g -DWITH_NAVIER_STOKES $(VERSION_FLAGS)

navier_stokes_hpc: ../feltor/feltor.cpp ../feltor/init.h navier_stokes.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DWITHOUT_GLFW -DWITH_NAVIER_STOKES $(VERSION_FLAGS)

.PHONY: clean

clean:
	rm -f navier_stokes navier_stokes_hpc
