device = omp

#configure machine 
include ../config/default.mk
include ../config/*.mk 
include ../config/devices/devices.mk

INCLUDE+= -I../inc/# include files libs
INCLUDE+= -I../src/# include files from source code

TARGETS =  toefl_video toefl_analytics toefl_compare show com figure17 vmax toefl_radial ncdiag fftwdiag crosscoherencdiag feltorSdiag feltorSHdiag

all: $(TARGETS)


toefl_video: toefl_video.cu 
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) -lhdf5 -lhdf5_hl 
	echo "Attention: No compiler warnings are issued!" 

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
fftwdiag: fftwdiag.cpp
	$(CC) $(OPT) $(CFLAGS) -std=c++0x $< -o $@ $(INCLUDE) $(LIBS)  
histdiag: histdiag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g
crosscoherencdiag: crosscoherencdiag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g
feltorSdiag: feltorSdiag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g
feltorSHdiag: feltorSHdiag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g

probe: probe.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) -lhdf5 -lhdf5_hl 

figure17: figure17.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS)

vmax: vmax.cu
	$(CC) $(OPT) $(CFLAGS) $< -o $@ -std=c++0x $(INCLUDE) 

.PHONY: clean

clean:
	rm -f $(TARGETS)
