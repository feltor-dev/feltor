device = omp

#configure machine 
include ../config/default.mk
include ../config/*.mk 
include ../config/devices/devices.mk

INCLUDE+= -I../inc/# include files libs
INCLUDE+= -I../src/# include files from source code

TARGETS =  toefl_video toefl_analytics toefl_compare show com figure17 vmax toefl_radial ncdiag filamentdiag fftwdiag crosscoherencdiag feltorSesoldiag feltorShwdiag feltorSHdiag feltordiag vmaxnc feltorSHvmaxdiag feltorSHvmaxisodiag feltorSHcompdiag feltorSHcompalldiag feltorSHvCOMdiag feltorSHvCOMisodiag feltorSHcompallvmaxdiag feltorSHcompallisovmaxdiag toeflRdiag

all: $(TARGETS)


toefl_video: toefl_video.cu 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) -lhdf5 -lhdf5_hl 

toefl_analytics: toefl_analytics.cu 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) -lhdf5 -lhdf5_hl   
toefl_compare: toefl_compare.cu 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) -lhdf5 -lhdf5_hl 
toefl_radial: toefl_radial.cu 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) -lhdf5 -lhdf5_hl   

show: show.cpp
	$(CC) $< -o $@ -std=c++0x $(CFLAGS) $(INCLUDE) -lhdf5 -lhdf5_hl -Wall

com: com.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS)

ncdiag: ncdiag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g
filamentdiag: filamentdiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) 
feltordiag: feltordiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) 
feltorSHdiag: feltorSHdiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_BENCHMARK
toeflRdiag: toeflRdiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB)
feltorSHvmaxdiag: feltorSHvmaxdiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS)
feltorSHvmaxisodiag: feltorSHvmaxisodiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS)
feltorSHvCOMdiag: feltorSHvCOMdiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS)
feltorSHvCOMisodiag: feltorSHvCOMisodiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS)
feltorSHcompdiag: feltorSHcompdiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS)
feltorSHcompalldiag: feltorSHcompalldiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS)
feltorSHcompallvmaxdiag: feltorSHcompallvmaxdiag.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS)
feltorSHcompallvmaxisodiag: feltorSHcompallvmaxisodiag.cu
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
# feltorSHdiag: feltorSHdiag.cpp
# 	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g

probe: probe.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) -lhdf5 -lhdf5_hl 

figure17: figure17.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS)

vmax: vmax.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ -std=c++0x $(INCLUDE) 

.PHONY: clean

clean:
	rm -f $(TARGETS)
