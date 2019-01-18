device=gpu

#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: feltor_hpc manufactured

manufactured: manufactured.cu manufactured.h feltor.cuh
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(JSONLIB)

feltor: feltor.cu feltor.cuh
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(JSONLIB) -g

feltor_hpc: feltor_hpc.cu feltor.cuh
	$(CC) -g $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB)

feltor_mpi: feltor_hpc.cu feltor.cuh
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DFELTOR_MPI

.PHONY: clean

clean:
	rm -f feltor feltor_hpc feltor_mpi manufactured
