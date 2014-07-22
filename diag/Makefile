INCLUDE = -I../inc/# include files libs
INCLUDE+= -I../src/# include files from source code
<<<<<<< HEAD
device = cpu
system = leo3
=======
device = gpu
system = home
>>>>>>> 08f521beb571a691e1102a0c2181c2f9767d111e

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
INCLUDE += -I/home/matthias/include # cusp and thrust libraries
#you might check the libs here, cf your glfw installation
GLFLAGS  = -lglfw3 -lXxf86vm -lXext -lX11 -lGLU  -lGL -lpthread -lm -lXi -lXrandr
LIBS 	 = -lhdf5 -lhdf5_hl # hdf5 libraries
endif

ifeq ($(strip $(system)),leo3)
CXX = mpicxx
INCLUDE += -I$(HOME)/include
INCLUDE += -I$(UIBK_HDF5_INC)
INCLUDE += -I$(UIBK_OPENMPI_INC)

LIBS 	+= -L$(UIBK_HDF5_LIB) -lhdf5 -lhdf5_hl
GLFLAGS  = -lm
endif

ifeq ($(strip $(system)),vsc)
CXX = g++
INCLUDE += -I$(HOME)/include
INCLUDE += -I/opt/sw/hdf5/1.8.11/include
INCLUDE += -I/opt/intel/impi/4.1.0/include64 
CFLAGS  += -DMPICH_IGNORE_CXX_SEEK

LIBS     = -L/opt/intel/composerxe/lib/intel64 -lirc -lsvml
LIBS    += -L/opt/intel/impi/4.1.0/lib64 -lmpi
LIBS 	+= -L/opt/sw/hdf5/1.8.11/lib -lhdf5 -lhdf5_hl -lz
GLFLAGS  = -lm
endif

TARGETS =  toefl_video toefl_analytics toefl_compare show com imp vmax toefl_radial feltor_video 

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
innto_video: innto_video.cpp
	$(CC) -O3  $< -o $@ -std=c++0x $(INCLUDE) $(GLFLAGS) -lhdf5 -lhdf5_hl -lfftw3 
innto_radial: innto_radial.cpp
	$(CC) -O3  $< -o $@ -std=c++0x $(INCLUDE) $(GLFLAGS) -lhdf5 -lhdf5_hl -lfftw3 
feltor_video: feltor_video.cu 
	$(CC) -O2 $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) -lhdf5 -lhdf5_hl 

show: show.cpp
	$(CXX) $< -o $@ -std=c++0x $(CFLAGS) $(INCLUDE) -lhdf5 -lhdf5_hl -Wall

com: com.cu
	$(CC) -O2 $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS)
<<<<<<< HEAD
=======

imp: imp.cu
	$(CC) -O2 $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS)

probe: probe.cu
	$(CC) -O2 $(CFLAGS) $< -o $@ $(INCLUDE) -lhdf5 -lhdf5_hl 
>>>>>>> 08f521beb571a691e1102a0c2181c2f9767d111e

figure17: figure17.cu
	$(CC) -O2 $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS)

vmax: vmax.cu
	$(CXX) -O2 $(CFLAGS) $< -o $@ -std=c++0x $(CFLAGS) $(INCLUDE) 
.PHONY: clean

clean:
	rm -f $(TARGETS)
