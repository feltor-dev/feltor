device=gpu

#configure machine 
include ../../config/default.mk
include ../../config/*.mk 
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: toeflR toefl_hpc

toeflR: toeflR.cu toeflR.cuh 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) -DDG_BENCHMARK 

toefl_hpc: toefl_hpc.cu toeflR.cuh 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_BENCHMARK

generator: generator.cpp
	g++ generator.cpp -o generator -std=c++0x -Wall

.PHONY: clean

clean:
	rm -f toeflR toefl_hpc
