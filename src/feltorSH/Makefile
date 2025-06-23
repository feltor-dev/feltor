device=gpu

#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: feltor feltor_hpc feltor_mpi feltorSHdiag feltorSHdiag2 feltorSHvmaxdiag

%: %.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB)

feltor: feltor.cpp feltor.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(LIBS) $(JSONLIB) -DWITH_GLFW

feltor_hpc: feltor.cpp feltor.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB)

feltor_mpi: feltor.cpp feltor.h
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DWITH_MPI

.PHONY: clean


clean:
	rm -f feltor feltor_hpc feltor_mpi feltorSHdiag feltorSHdiag2 feltorSHvmaxdiag
