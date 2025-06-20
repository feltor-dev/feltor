device=gpu

#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: feltor feltor_hpc feltor_mpi crosscoherencdiag feltorShwdiag feltorShwradstat fftwdiag histdiag feltorShwmerger feltorShwstat growthrate normdiag

feltor: feltor.cpp feltor.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(LIBS) $(JSONLIB) -DWITH_GLFW

feltor_hpc: feltor.cpp feltor.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB)

feltor_mpi: feltor.cpp feltor.h
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DWITH_MPI

%: %.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB)

fftwdiag: fftwdiag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -I$(HOME)/include/spectral -lfftw3 -g
growthrate: growthrate.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -I$(HOME)/include/spectral -lfftw3 -g

.PHONY: clean


clean:
	rm -f feltor feltor_hpc feltor_mpi crosscoherencdiag feltorShwdiag feltorShwradstat fftwdiag histdiag feltorShwmerger feltorShwstat growthrate normdiag
