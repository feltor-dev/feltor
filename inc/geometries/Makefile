device=gpu

#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../../inc   # other project libraries

all: $(CPPFILES:%.cpp=%)

CATCH=../../tests/catch-tests
CATCHMPI=../../tests/catch-tests-mpi

CPPFILES=$(wildcard *.cpp)

TARGETS=modified_t\
solovev_t\
polynomial_t\

#TARGETSMPI=

#all: $(CPPFILES:%.cpp=%) $(CUFILES:%.cu=%)
all: tests mpi-tests

%_b: %_b.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) -g $(LIBS) $(JSONLIB)

%_mpib: %_b.cpp
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) -DWITH_MPI $(LIBS) $(JSONLIB)

%_t.$(device).o: %_t.cpp %.h
	$(CC) $(OPT) $(CFLAGS) -c $< -o $@ -g $(INCLUDE)

mpi_%_t.$(device).o: %_t.cpp
	$(MPICC) $(OPT) $(MPICFLAGS) -c $< -o $@ -g $(INCLUDE) -DWITH_MPI

%_t: $(CATCH).$(device).o %_t.$(device).o
	$(CC) $(OPT) -o $@ $^ -lCatch2 $(SPARSELIB) $(JSONLIB)

mpi_%_t: $(CATCHMPI).$(device).o mpi_%_t.$(device).o
	$(MPICC) $(OPT) -o $@ $^ -lCatch2 $(SPARSELIB) $(JSONLIB)

$(CATCH).$(device).o: $(CATCH).cpp
	$(CC) $(OPT) $(CFLAGS) $(INCLUDE) -g -c $< -o $@

$(CATCHMPI).$(device).o: $(CATCHMPI).cpp
	$(MPICC) $(OPT) $(MPICFLAGS) $(INCLUDE) -g -c $< -o $@

tests: $(CATCH).$(device).o $(addsuffix .$(device).o,$(TARGETS))
	$(CC) -o $@ $^ -lCatch2 $(SPARSELIB) $(JSONLIB)

mpi-tests: $(CATCHMPI).$(device).o $(addsuffix .$(device).o,$(TARGETSMPI))
	$(MPICC) -o $@ $^ -lCatch2 $(SPARSELIB) $(JSONLIB)

.PHONY: clean doc

doc:
	doxygen Doxyfile

clean:
	rm -f *.o *_t *_b *_mpit *_mpib tests mpi-tests
