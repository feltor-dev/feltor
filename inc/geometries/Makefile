device=gpu

#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../../inc   # other project libraries

CPPFILES=$(wildcard *.cpp)

all: $(CPPFILES:%.cpp=%)

%_t: %_t.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g $(SPARSELIB)

%_b: %_b.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g $(SPARSELIB)

%_mpit: %_t.cpp
	$(MPICC) $(OPT) $(INCLUDE) $(MPICFLAGS)  $< -o $@ $(LIBS) $(JSONLIB) -g -DWITH_MPI $(SPARSELIB)

%_mpib: %_b.cpp
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ -g $(INCLUDE) $(LIBS) $(JSONLIB) -g -DWITH_MPI $(SPARSELIB)



.PHONY: clean doc

doc:
	doxygen Doxyfile

clean:
	rm -f *_t *_b *_mpit *_mpib
