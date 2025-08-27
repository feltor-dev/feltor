device=gpu

#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../../ # other project libraries
INCLUDE+= -I../    # other project libraries

CATCH=../../tests/catch-tests
CATCHMPI=../../tests/catch-tests-mpi

CPPFILES=$(wildcard *.cpp)
CUFILES=$(wildcard *.cu)

TARGETS=functors_t\
tridiaginv_t\
exp_runge_kutta_t\
contours_t

all: $(CPPFILES:%.cpp=%) $(CUFILES:%.cu=%)

%_b: %_b.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LAPACKLIB) -g

%_mpib: %_mpib.cpp
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LAPACKLIB) -g

# CATCH2 TESTS
#
$(CATCH).$(device).o: $(CATCH).cpp
	$(CC) $(OPT) $(CFLAGS) $(INCLUDE) -g -c $< -o $@

%_t.$(device).o: %_t.cpp %.h
	$(CC) $(OPT) $(CFLAGS) -c $< -o $@ -g $(INCLUDE)

%_t: $(CATCH).$(device).o %_t.$(device).o
	$(CC) $(OPT) -o $@ $^ -lCatch2 $(SPARSELIB) $(LAPACKLIB)

tests: $(CATCH).$(device).o $(addsuffix .$(device).o,$(TARGETS))
	$(CC) -o $@ $^ -lCatch2 $(SPARSELIB)

.PHONY: clean doc

doc:
	doxygen Doxyfile

clean:
	rm -f *.o *_t *_b *_mpit *_mpib tests mpi-tests
