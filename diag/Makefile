device = omp

#configure machine 
include ../config/default.mk
include ../config/*.mk 
include ../config/devices/devices.mk

INCLUDE+= -I../inc/# include files libs
INCLUDE+= -I../src/# include files from source code

<<<<<<< HEAD

TARGETS =  toefl_video toefl_analytics toefl_compare show com figure17 vmax toefl_radial ncdiag filamentdiag fftwdiag crosscoherencdiag feltorSesoldiag feltorShwdiag feltorSHdiag feltordiag vmaxnc feltorSHvmaxdiag feltorSHvmaxisodiag feltorSHcompdiag feltorSHcompalldiag feltorSHvCOMdiag feltorSHvCOMisodiag feltorSHcompallvmaxdiag feltorSHcompallisovmaxdiag toeflRdiag impRdiag feltorShwinntodiag
=======
TARGETS =  compare histdiag ncdiag filamentdiag fftwdiag crosscoherencdiag feltorSesoldiag feltorShwdiag feltorSHdiag feltordiag vmaxnc feltorSHvmaxdiag toeflRdiag impRdiag
>>>>>>> de19e28d868789d715ec18f929a800bc31e1bcdc

all: $(TARGETS)


ncdiag: ncdiag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g
filamentdiag: filamentdiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) 
feltordiag: feltordiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -g
feltorSHdiag: feltorSHdiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_BENCHMARK
toeflRdiag: toeflRdiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB)
impRdiag: impRdiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g
feltorSHvmaxdiag: feltorSHvmaxdiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS)
vmaxnc: vmaxnc.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) 
fftwdiag: fftwdiag.cpp
	$(CC) $(OPT) $(CFLAGS) -std=c++0x $< -o $@ $(INCLUDE) $(LIBS)  -lfftw3
histdiag: histdiag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g
compare: compare.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g
crosscoherencdiag: crosscoherencdiag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g
feltorSesoldiag: feltorSesoldiag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g
feltorShwdiag: feltorShwdiag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g
<<<<<<< HEAD
feltorShwinntodiag: feltorShwinntodiag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g
# feltorSHdiag: feltorSHdiag.cpp
# 	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g

probe: probe.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) -lhdf5 -lhdf5_hl 

figure17: figure17.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS)

vmax: vmax.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ -std=c++0x $(INCLUDE) 
=======
>>>>>>> de19e28d868789d715ec18f929a800bc31e1bcdc

.PHONY: clean

clean:
	rm -f $(TARGETS)
