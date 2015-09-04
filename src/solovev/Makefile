device=gpu

#configure machine 
include ../../config/default.mk
include ../../config/*.mk 
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: ds_geom_t geometry_t geometry_diag

ds_geom_t: ds_geom_t.cu geometry.h 
	$(CC) -O2 $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE) $(LIBS)  -DDGDEBUG -DDG_BENCHMARK 

geometry_t: geometry_t.cu geometry.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE) 
	
geometry_diag: geometry_diag.cu geometry.h 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(LIBS) $(INCLUDE) 

.PHONY: clean doc

doc: 
	doxygen Doxyfile
	
clean:
	rm -f geometry_diag geometry_t ds_geom_t
