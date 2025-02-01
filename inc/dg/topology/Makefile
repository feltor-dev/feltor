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

all: $(CPPFILES:%.cpp=%) $(CUFILES:%.cu=%)

backscatter_t: backscatter_t.cu xspacelib.h
	$(CC) $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE) -g

%_mpib: %_mpib.cu
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE)

%_b: %_b.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) -g

%_t.$(device).o: %_t.cu %.h
	$(CC) $(OPT) $(CFLAGS) -c $< -o $@ -g $(INCLUDE)

%_mpit: %_mpit.cu
	$(MPICC) $(OPT) $(INCLUDE) $(MPICFLAGS) $< -o $@ -g
#%_mpit.$(device).o: %_t.cu %.h
#	$(MPICC) $(OPT) $(MPICFLAGS) -c $< -o $@ -g $(INCLUDE) -DWITH_MPI

%_t: $(CATCH).$(device).o %_t.$(device).o
	$(CC) $(OPT) -o $@ $^

#%_mpit: $(CATCHMPI).$(device).o %_mpit.$(device).o
#	$(MPICC) $(OPT) -o $@ $^

$(CATCH).$(device).o: $(CATCH).cpp
	$(CC) $(OPT) $(CFLAGS) $(INCLUDE) -g -c $< -o $@

#$(CATCHMPI).$(device).o: $(CATCHMPI).cpp
#	$(MPICC) $(OPT) $(MPICFLAGS) $(INCLUDE) -g -c $< -o $@

tests: $(CATCH).$(device).o $(addsuffix .$(device).o,$(TARGETS))
	$(CC) -o $@ $^

#tests-mpi: $(CATCHMPI).$(device).o $(addsuffix .$(device).o,$(TARGETSMPI))
#	$(MPICC) -o $@ $^

.PHONY: clean

clean:
	rm -f *.o *_t *_b *_mpit *_mpib tests tests-mpi
