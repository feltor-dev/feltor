device=gpu

#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../../inc   # other project libraries

CUFILES=$(wildcard *.cu)

all: $(CUFILES:%.cu=%)

geometry_diag: geometry_diag.cu solovev.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(LIBS) $(INCLUDE) $(JSONLIB) -g

%_t: %_t.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g

%_b: %_b.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g

%_mpit: %_mpit.cu
	$(MPICC) $(OPT) $(INCLUDE) $(MPICFLAGS)  $< -o $@ $(LIBS) $(JSONLIB) -g

%_mpib: %_mpib.cu
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ -g $(INCLUDE) $(LIBS) $(JSONLIB) -g



.PHONY: clean doc

doc:
	doxygen Doxyfile

clean:
	rm -f geometry_diag *_t *_b *_mpit *_mpib
