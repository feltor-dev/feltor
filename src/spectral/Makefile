system = leo3
INCLUDE = -I../../inc

CFLAGS = -Wall -std=c++0x -fopenmp
LIBS = -lfftw3 -lm -lGL -lglfw -lrt -lX11 -lXxf86vm
CXX = g++

ifeq ($(strip $(system)),leo3)
INCLUDE += -I$(HOME)/include
INCLUDE += -I$(UIBK_HDF5_INC)
#INCLUDE += -I$(UIBK_OPENMPI_INC)
INCLUDE += -I$(UIBK_FFTW_INC)

LIBS 	 = -L$(UIBK_HDF5_LIB) -lhdf5 -lhdf5_hl 
LIBS    += -L$(UIBK_FFTW_LIB) -lfftw3 
GLFLAGS  = -lm
CXX = mpicxx
endif


all: innto innblobs innto_hpc

innto: innto.cpp dft_dft_solver.h drt_dft_solver.h blueprint.h equations.h
	$(CXX) -O3 $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@

innblobs: innblobs.cpp dft_dft_solver.h drt_dft_solver.h blueprint.h equations.h
	$(CXX) -O3 $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@

innto_t: innto.cpp dft_dft_solver.h drt_dft_solver.h blueprint.h equations.h
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@

innto_hpc: innto_hpc.cpp drt_dft_solver.h blueprint.h equations.h
	$(CXX) -O3 $< $(CFLAGS) -o $@ $(INCLUDE) $(LIBS) 


%_t: %_t.cpp %.h
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@

%_b: %_t.cpp %.h
	$(CXX) -O3  $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@
clean:
	rm -f *_t *_b innto innblobs innto_hpc
