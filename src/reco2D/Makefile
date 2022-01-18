device=gpu

#configure machine
include ../../config/default.mk
include ../../config/version.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: reconnection reconnection_hpc reconnection_mpi

reconnection: reconnection.cu reconnection.h init.h diag.h parameters.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(LIBS) $(JSONLIB) -DDG_BENCHMARK $(VERSION_FLAGS) -g

reconnection_hpc: reconnection.cu reconnection.h init.h diag.h parameters.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DDG_BENCHMARK $(VERSION_FLAGS) -DWITHOUT_GLFW

reconnection_mpi: reconnection.cu reconnection.h init.h diag.h
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DWITH_MPI -DDG_BENCHMARK $(VERSION_FLAGS) -DWITHOUT_GLFW

.PHONY: clean

clean:
	rm -f reconnection reconnection_hpc reconnection_mpi
