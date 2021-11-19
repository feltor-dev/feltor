device = omp

#configure machine
include ../config/default.mk
include ../config/*.mk
include ../config/devices/devices.mk

INCLUDE+= -I../inc/# include files libs
INCLUDE+= -I../src/# include files from source code

TARGETS =  compare histdiag fftwdiag crosscoherencdiag feltorSesoldiag feltorShwdiag feltorSHdiag vmaxnc feltorSHvmaxdiag toeflRdiag impRdiag feltorShwmerger feltorShwradstat feltorShwstat growthrate toeflEPdiag normdiag

all: $(TARGETS)

feltorSHdiag: feltorSHdiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB)
toeflRdiag: toeflRdiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB)
toeflEPdiag: toeflEPdiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB)
impRdiag: impRdiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g
feltorSHvmaxdiag: feltorSHvmaxdiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB)
vmaxnc: vmaxnc.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS)
fftwdiag: fftwdiag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -I$(HOME)/include/spectral -lfftw3  -DTL_DEBUG -g
growthrate: growthrate.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -I$(HOME)/include/spectral -lfftw3  -DTL_DEBUG -g
histdiag: histdiag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g
compare: compare.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g
crosscoherencdiag: crosscoherencdiag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g $(JSONLIB)
feltorSesoldiag: feltorSesoldiag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g $(JSONLIB)
feltorShwdiag: feltorShwdiag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DDG_DEBUG -g
feltorShwmerger: feltorShwmerger.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DDG_DEBUG -g
feltorShwstat: feltorShwstat.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DDG_DEBUG -g
feltorShwradstat: feltorShwradstat.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DDG_DEBUG -g
normdiag: normdiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DDG_DEBUG -g
reco2Ddiag: reco2Ddiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DDG_DEBUG -g
.PHONY: clean

clean:
	rm -f $(TARGETS)

