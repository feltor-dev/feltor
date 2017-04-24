device=gpu

#configure machine 
include ../../config/default.mk
include ../../config/*.mk 
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: asela

asela: asela.cu asela.cuh 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(JSONLIB) -DDG_BENCHMARK 

.PHONY: clean

doc: 
	doxygen Doxyfile
	
clean:
	rm -f asela 
