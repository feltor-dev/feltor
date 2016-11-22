device=gpu

#configure machine 
include ../../config/default.mk
include ../../config/*.mk 
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: asela asela_hpc

asela: asela.cu asela.cuh 
	$(CC) $(OPT) $(CFLAGS) $< -o $@  -g $(INCLUDE) $(GLFLAGS) -DDG_BENCHMARK 

asela_hpc: asela_hpc.cu asela.cuh 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_BENCHMARK

asela_mpi: asela_mpi.cu asela.cuh 
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_BENCHMARK
	
.PHONY: clean doc

doc: 
	doxygen Doxyfile
	
clean:
	rm -f asela asela_hpc asela_mpi
