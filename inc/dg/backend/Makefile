device=gpu

#configure machine
include ../../../config/default.mk
include ../../../config/*.mk
include ../../../config/devices/devices.mk

INCLUDE+= -I../../ # other project libraries
INCLUDE+= -I../    # other project libraries

TARGETS=fma_t\
gather_t\
index_t\
memory_t\
traits_t\
view_t

TARGETSMPI=mpi_gather_kron_mpit\
mpi_gather_mpit\
mpi_init_mpit\
mpi_kron_mpit\
mpi_exblas_mpit\
mpi_permutation_mpit

CATCH=../../../tests/catch-tests
CATCHMPI=../../../tests/catch-tests-mpi

CPPFILES=$(wildcard *.cpp)
CUFILES=$(wildcard *.cu)

all: $(CPPFILES:%.cpp=%) $(CUFILES:%.cu=%)

%_t.$(device).o: %_t.cpp
	$(CC) $(OPT) $(CFLAGS) -c $< -o $@ -g $(INCLUDE)

%_mpit.$(device).o: %_mpit.cpp
	$(MPICC) $(OPT) $(INCLUDE) $(MPICFLAGS) -c $< -o $@ -g

version: version.cpp
	$(CC) $(CFLAGS) $(INCLUDE) $< -o $@
	./$@

%_t: $(CATCH).$(device).o %_t.$(device).o
	$(CC) $(OPT) -o $@ $^ -lCatch2

%_mpit: $(CATCHMPI).$(device).o %_mpit.$(device).o
	$(MPICC) $(OPT) -o $@ $^ -lCatch2

$(CATCH).$(device).o: $(CATCH).cpp
	$(CC) $(OPT) $(CFLAGS) $(INCLUDE) -g -c $< -o $@

$(CATCHMPI).$(device).o: $(CATCHMPI).cpp
	$(MPICC) $(OPT) $(MPICFLAGS) $(INCLUDE) -g -c $< -o $@

tests: $(CATCH).$(device).o $(addsuffix .$(device).o,$(TARGETS))
	$(CC) -o $@ $^ -lCatch2

mpi-tests: $(CATCHMPI).$(device).o $(addsuffix .$(device).o,$(TARGETSMPI))
	$(MPICC) -o $@ $^ -lCatch2

.PHONY: clean

clean:
	rm -f *.o *_t *_mpit version tests mpi-tests
