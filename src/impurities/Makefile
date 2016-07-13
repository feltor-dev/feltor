device=gpu

#configure machine 
include ../../config/default.mk
include ../../config/*.mk 
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: toeflI toefl_hpc

toeflI: toeflI.cu toeflI.cuh 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(JSONLIB) -DDG_BENCHMARK -g

toefl_hpc: toefl_hpc.cu toeflI.cuh 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB)  -DDG_BENCHMARK -g

toefl_mpi: toefl_mpi.cu toeflI.cuh
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DDG_BENCHMARK -g

.PHONY: clean

clean:
	rm -f toeflI toefl_hpc toefl_mpi
