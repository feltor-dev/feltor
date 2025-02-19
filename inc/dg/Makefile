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

TARGETS=blas_t\
blas1_t\
pcg_t\
simpsons_t\
extrapolation_t\
nullstelle_t\
runge_kutta_t\
adaptive_t\
multistep_t\
arakawa_t\

TARGETSMPI=mpi_average_t\
mpi_arakawa_t\
mpi_blas_t\
mpi_blas1_t\

all: $(CPPFILES:%.cpp=%) $(CUFILES:%.cu=%)

bathRZ_t: bathRZ_t.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE)  -g

backscatter_t: backscatter_t.cpp xspacelib.h
	$(CC) $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE) -g

cluster_mpib: cluster_mpib.cu
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(JSONLIB) -g


%_mpib: %_b.cpp
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) -DWITH_MPI

%_b: %_b.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) -g

%_t.$(device).o: %_t.cpp %.h
	$(CC) $(OPT) $(CFLAGS) -c $< -o $@ -g $(INCLUDE)

mpi_%_t.$(device).o: %_t.cpp
	$(MPICC) $(OPT) $(MPICFLAGS) -c $< -o $@ -g $(INCLUDE) -DWITH_MPI


%_mpit.$(device).o: %_mpit.cpp %.h
	$(MPICC) $(OPT) $(MPICFLAGS) -c $< -o $@ -g $(INCLUDE) -DWITH_MPI

%_t: $(CATCH).$(device).o %_t.$(device).o
	$(CC) $(OPT) -o $@ $^ -lCatch2

mpi_%_t: $(CATCHMPI).$(device).o mpi_%_t.$(device).o
	$(MPICC) $(OPT) -o $@ $^ -lCatch2

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

.PHONY: clean doc

doc:
	doxygen Doxyfile

clean:
	rm -f *.o *_t *_b *_mpit *_mpib tests mpi-tests
