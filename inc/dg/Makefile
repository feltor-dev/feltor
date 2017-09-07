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


%_mpit: %_mpit.cu 
	$(MPICC) $(INCLUDE) -DDG_DEBUG $(MPICFLAGS) $< -o $@ -g

%_mpib: %_mpib.cu
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) -g

%_t: %_t.cu
	$(CC) $(OPT) $(INCLUDE) -DDG_DEBUG $(CFLAGS) $< -o $@ 

%_b: %_b.cu 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) -g 

bathRZ_t: bathRZ_t.cu 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(GLFLAGS) $(INCLUDE) 
	
.PHONY: clean doc

doc: 
	doxygen Doxyfile

clean:
	rm -f *_t *_b *_mpit *_mpib 
