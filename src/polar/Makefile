device=omp
OS=arch

#configure machine 
include ../../config/default.mk
include ../../config/*.mk 
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries


all: polar polargl

polar: polar.cu ns.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@  $(INCLUDE) $(GLFLAGS) $(JSONLIB)
polargl: polar.cu ns.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@  $(INCLUDE) $(GLFLAGS) $(JSONLIB) -DOPENGL_WINDOW
polargl: polar.cu ns.h
.PHONY: clean

clean:
	rm -f polar
