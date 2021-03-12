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
	$(CC) $(OPT)$(INCLUDE)  -DDG_DEBUG $(CFLAGS) $< -o $@   $(LIBS)  -g

%_b: %_b.cu
	$(CC) $(OPT) $(CFLAGS) -DDG_BENCHMARK $< -o $@ $(INCLUDE) $(LIBS) -g

%_mpit: %_mpit.cu
	$(MPICC) $(OPT) $(INCLUDE) -DDG_DEBUG $(MPICFLAGS) $< -o $@ -g

%_mpib: %_mpib.cu
	$(MPICC) $(OPT) $(MPICFLAGS) -DDG_BENCHMARK $< -o $@ $(INCLUDE) -g

bathRZ_t: bathRZ_t.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE)  -g

.PHONY: clean doc

doc:
	doxygen Doxyfile

clean:
	rm -f *_t *_b *_mpit *_mpib
