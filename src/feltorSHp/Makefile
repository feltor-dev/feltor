device=gpu

#configure machine 
include ../../config/default.mk
include ../../config/*.mk 
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: feltor feltor_hpc feltor_log 

feltor: feltor.cu feltor.cuh 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(JSONLIB)-DDG_BENCHMARK 

feltor_log: feltor_log.cu feltor_log.cuh 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(JSONLIB)-DDG_BENCHMARK 	
	
feltor_hpc: feltor_hpc.cu feltor.cuh 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DDG_BENCHMARK

feltor_mpi: feltor_mpi.cu feltor.cuh 
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DDG_BENCHMARK
	
generator: generator.cpp 
	 $(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) -std=c++0x -Wall


.PHONY: clean doc

doc: 
	doxygen Doxyfile
	
clean:
	rm -f feltor feltor_hpc feltor_mpi
