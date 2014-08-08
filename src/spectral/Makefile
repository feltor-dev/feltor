system = home
INCLUDE = -I../../inc
INCLUDE += -I$(HOME)/include

CFLAGS = -Wall -std=c++0x -fopenmp
GLFLAGS= -lfftw3 -lm -lGL -lglfw3 -lrt -lX11 -lXxf86vm -lXi -lXrandr
LIBS = -lfftw3 -lhdf5 -lhdf5_hl

ifeq ($(strip $(system)),leo3)
INCLUDE += -I$(UIBK_HDF5_INC)
#INCLUDE += -I$(UIBK_OPENMPI_INC)
INCLUDE += -I$(UIBK_FFTW_INC)

LIBS 	 = -L$(UIBK_HDF5_LIB) -lhdf5 -lhdf5_hl 
LIBS    += -L$(UIBK_FFTW_LIB) -lfftw3 
GLFLAGS  = -lm
CXX = mpicxx
endif


all: innto innblobs innto_hpc innto_hw

innto: innto.cpp dft_dft_solver.h drt_dft_solver.h blueprint.h equations.h
	$(CXX) -O3 $< $(CFLAGS) $(INCLUDE) $(GLFLAGS) -o $@

innto_per: innto_per.cpp dft_dft_solver.h blueprint.h equations.h
	$(CXX) -O3 $< $(CFLAGS) $(INCLUDE) $(GLFLAGS) -o $@

innblobs: innblobs.cpp dft_dft_solver.h drt_dft_solver.h blueprint.h equations.h
	$(CXX) -O3 $< $(CFLAGS) $(INCLUDE) $(GLFLAGS) -o $@

innto_t: innto.cpp dft_dft_solver.h drt_dft_solver.h blueprint.h equations.h
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(GLFLAGS) -o $@
	./$@

innto_hpc: innto_hpc.cpp drt_dft_solver.h blueprint.h equations.h
	$(CXX) -O3 $< $(CFLAGS) -o $@ $(INCLUDE) $(LIBS) 

innto_hw: innto_hw.cpp dft_dft_solver.h blueprint.h equations.h energetics.h
	$(CXX) -O2 $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@

energetics_t: energetics_t.cpp energetics.h
	$(CXX) -O2 $< $(CFLAGS) $(INCLUDE) -lm -lfftw3 -o $@

%_t: %_t.cpp %.h
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@

%_b: %_t.cpp %.h
	$(CXX) -O3  $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@

generator: generator.cpp
	g++ generator.cpp -o generator -std=c++0x
generator_hw: generator_hw.cpp
	g++ generator_hw.cpp -o generator_hw -std=c++0x
	./generator_hw

clean:
	rm -f *_t *_b innto innblobs innto_hpc
