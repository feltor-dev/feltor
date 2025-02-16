device=gpu

#configure machine
include ../../../config/default.mk
include ../../../config/*.mk
include ../../../config/devices/devices.mk

INCLUDE+= -I../../    # other project libraries

CATCH=../../../tests/catch-tests
CATCHMPI=../../../tests/catch-tests-mpi

CPPFILES=$(wildcard *.cpp)
CUFILES=$(wildcard *.cu)

TARGETS=average_t\
base_geometry_t\
derivatives_t\
derivativesX_t\
dlt_t\
dx_t\
evaluation_t\
fem_t\
filter_t\
geometry_t\
grid_t\
interpolation_t\
multiply_t\
operator_t\
operator_tensor_t\
projection_t\
refined_grid_t\
stencil_t\
tensor_t\
transform_t\

TARGETSMPI=mpi_average_t\
mpi_evaluation_t\
mpi_dx_t\
mpi_derivatives_t\
mpi_average_t\
mpi_grid_mpit\
filter_mpit\
stencil_mpit\
interpolation_mpit\
mpi_projection_t\

#all: $(CPPFILES:%.cpp=%) $(CUFILES:%.cu=%)
all: tests mpi-tests

backscatter_t: backscatter_t.cpp xspacelib.h
	$(CC) $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE) -g

%_t.$(device).o: %_t.cpp %.h
	$(CC) $(OPT) $(CFLAGS) -c $< -o $@ -g $(INCLUDE)

mpi_%_t.$(device).o: %_t.cpp mpi_%.h
	$(MPICC) $(OPT) $(MPICFLAGS) -c $< -o $@ -g $(INCLUDE) -DWITH_MPI

# There is no mpi_dx.h
mpi_dx_t.$(device).o: dx_t.cpp mpi_derivatives.h derivativesT.h
	$(MPICC) $(OPT) $(MPICFLAGS) -c $< -o $@ -g $(INCLUDE) -DWITH_MPI
# There is no mpi_average.h
mpi_average_t.$(device).o: average_t.cpp average.h
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

.PHONY: clean

clean:
	rm -f *.o *_t *_mpit tests mpi-tests
