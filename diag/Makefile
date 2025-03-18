device = omp

#configure machine
include ../config/default.mk
include ../config/*.mk
include ../config/devices/devices.mk

INCLUDE+= -I../inc/# include files libs
INCLUDE+= -I../src/# include files from source code

TARGETS =  compare histdiag fftwdiag crosscoherencdiag feltorSesoldiag feltorShwdiag feltorSHdiag feltorSHvmaxdiag feltorShwmerger feltorShwradstat feltorShwstat growthrate normdiag

all: $(TARGETS)

feltorSHdiag: feltorSHdiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB)
feltorSHvmaxdiag: feltorSHvmaxdiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB)
fftwdiag: fftwdiag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -I$(HOME)/include/spectral -lfftw3 -g
growthrate: growthrate.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -I$(HOME)/include/spectral -lfftw3 -g
histdiag: histdiag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g
compare: compare.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g
crosscoherencdiag: crosscoherencdiag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g
feltorSesoldiag: feltorSesoldiag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g
feltorShwdiag: feltorShwdiag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g
feltorShwmerger: feltorShwmerger.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g
feltorShwstat: feltorShwstat.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g
feltorShwradstat: feltorShwradstat.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g
normdiag: normdiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g
reco2Ddiag: reco2Ddiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g
.PHONY: clean

clean:
	rm -f $(TARGETS)

