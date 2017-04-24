device=omp
OS=arch

#configure machine 
include ../../config/default.mk
include ../../config/*.mk 
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries


all: polar polargl logpolar logpolargl

polar: polar.cu ns.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@  $(INCLUDE) $(JSONLIB)
polargl: polar.cu ns.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@  $(INCLUDE) $(GLFLAGS) $(JSONLIB) -DOPENGL_WINDOW
logpolar: polar.cu ns.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@  $(INCLUDE) $(JSONLIB) -DLOG_POLAR
logpolargl: polar.cu ns.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@  $(INCLUDE) $(GLFLAGS) $(JSONLIB) -DOPENGL_WINDOW -DLOG_POLAR
mpi: polar_mpi.cu ns.h
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@  $(INCLUDE) $(JSONLIB)
.PHONY: clean

clean:
	rm -f polar polargl logpolar logpolargl
