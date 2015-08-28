system = home
device = omp

INCLUDE = -I../inc/# include files libs
INCLUDE+= -I../src/# include files from source code

ifeq ($(strip $(device)),gpu)
CC = nvcc
CFLAGS = --compiler-options -Wall -arch=sm_20 -w
CFLAGS+= -Xcompiler -fopenmp 
#device system = cuda
CFLAGS+= -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP
OPT	   = -O2
else
CC = mpicxx
CFLAGS = -Wall -fopenmp -x c++
CFLAGS+= -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP
CFLAGS+= -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
OPT	   = -O3
endif

ifeq ($(strip $(system)),home)
CXX = g++
CXXFLAGS = -Wall -x c++
CXXFLAGS+= -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
INCLUDE += -I$(HOME)/include # cusp and thrust libraries
#you might check the libs here, cf your glfw installation
GLFLAGS  = -lglfw3 -lXxf86vm -lXext -lX11 -lGLU  -lGL -lpthread -lm -lXi -lXrandr
LIBS 	 = -lfftw3 -lnetcdf -lhdf5 -lhdf5_hl # hdf5 libraries
endif

ifeq ($(strip $(system)),leo3)
CXX = mpicxx
INCLUDE += -I$(HOME)/include
INCLUDE += -I$(UIBK_HDF5_INC)
INCLUDE += -I$(UIBK_OPENMPI_INC)
INCLUDE += -I$(UIBK_NETCDF_4_INC)

LIBS 	 = -L$(UIBK_HDF5_LIB) -lhdf5 -lhdf5_hl
LIBS 	+= -L$(UIBK_NETCDF_4_LIB) -lnetcdf -lcurl -lm 
GLFLAGS  = -lm
endif

ifeq ($(strip $(system)),vsc)
CXX = g++
INCLUDE += -I$(HOME)/include
INCLUDE += -I/opt/sw/hdf5/1.8.11/include
INCLUDE += -I/opt/intel/impi/4.1.0/include64 
INCLUDE += -I/opt/sw/netcdf/4.3/include
CFLAGS  += -DMPICH_IGNORE_CXX_SEEK

LIBS     = -L/opt/intel/composerxe/lib/intel64 -lirc -lsvml
LIBS    += -L/opt/intel/impi/4.1.0/lib64 -lmpi
LIBS 	+= -L/opt/sw/hdf5/1.8.11/lib -lhdf5 -lhdf5_hl -lz
LIBS 	+= -L/opt/sw/netcdf/4.3/lib -lnetcdf
GLFLAGS  = -lm
endif

TARGETS =  toefl_video toefl_analytics toefl_compare show com imp vmax toefl_radial feltor_video ncdiag fftwdiag crosscoherencdiag feltorSdiag feltorSHdiag

all: $(TARGETS)


toefl_video: toefl_video.cu 
	$(CC) -O2 $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) -lhdf5 -lhdf5_hl 
	echo "Attention: No compiler warnings are issued!" 

toefl_analytics: toefl_analytics.cu 
	$(CC) -O2 $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) -lhdf5 -lhdf5_hl   
toefl_compare: toefl_compare.cu 
	$(CC) -O2 $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) -lhdf5 -lhdf5_hl 
toefl_radial: toefl_radial.cu 
	$(CC) -O2 $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) -lhdf5 -lhdf5_hl   

show: show.cpp
	$(CXX) $< -o $@ -std=c++0x $(CXXFLAGS) $(INCLUDE) -lhdf5 -lhdf5_hl -Wall

com: com.cu
	$(CXX) -O2 $(CXXFLAGS) $< -o $@ $(INCLUDE) $(LIBS)

ncdiag: ncdiag.cpp
	$(CXX) -O2 $(CXXFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g
fftwdiag: fftwdiag.cpp
	$(CXX) -O2 $(CXXFLAGS) -std=c++0x $< -o $@ $(INCLUDE) $(LIBS)  
histdiag: histdiag.cpp
	$(CXX) -O2 $(CXXFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g
crosscoherencdiag: crosscoherencdiag.cpp
	$(CXX) -O2 $(CXXFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g
feltorSdiag: feltorSdiag.cpp
	$(CXX) -O2 $(CXXFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g
feltorSHdiag: feltorSHdiag.cpp
	$(CXX) -O2 $(CXXFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_DEBUG -g
imp: imp.cu
	$(CXX) -O2 $(CXXFLAGS) $< -o $@ $(INCLUDE) $(LIBS)

probe: probe.cu
	$(CXX) -O2 $(CXXFLAGS) $< -o $@ $(INCLUDE) -lhdf5 -lhdf5_hl 

figure17: figure17.cu
	$(CC) -O2 $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS)

vmax: vmax.cu
	$(CXX) -O2 $(CXXFLAGS) $< -o $@ -std=c++0x $(INCLUDE) 

.PHONY: clean

clean:
	rm -f $(TARGETS)
