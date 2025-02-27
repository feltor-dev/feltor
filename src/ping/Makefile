device=gpu

#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

mpi-ping: mpi-ping.cpp
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ -g $(INCLUDE)

.PHONY: clean

clean:
	rm -f mpi-ping
