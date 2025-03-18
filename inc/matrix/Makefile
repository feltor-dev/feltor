device=gpu

#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../../ # other project libraries
INCLUDE+= -I../    # other project libraries

CPPFILES=$(wildcard *.cpp)
CUFILES=$(wildcard *.cu)

all: $(CPPFILES:%.cpp=%) $(CUFILES:%.cu=%)

%_t: %_t.cu
	$(CC) $(OPT)$(INCLUDE) $(CFLAGS) $< -o $@   $(LAPACKLIB)  -g

%_b: %_b.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LAPACKLIB) -g

%_mpib: %_mpib.cu
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LAPACKLIB) -g

polarization_init_t: polarization_init_t.cu polarization_init.h
	$(CC) $(OPT)$(INCLUDE) $(CFLAGS) $< -o $@   $(LAPACKLIB) $(LIBS) -g

matrixprodexp_t: matrixprodexp_t.cu 
	$(CC) $(OPT)$(INCLUDE) $(CFLAGS) $< -o $@   $(LAPACKLIB) $(LIBS) -g
.PHONY: clean doc

doc:
	doxygen Doxyfile

clean:
	rm -f *_t *_b *_mpit *_mpib
