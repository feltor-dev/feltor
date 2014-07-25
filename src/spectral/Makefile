INCLUDE += -I../../inc
INCLUDE += -I/home/markus/Dokumente/Phd/Include
CFLAGS = -Wall -std=c++0x -fopenmp
LIBS = -lfftw3 -lm -lGL -lglfw3 -lrt -lX11 -lXxf86vm -lXi -lXrandr
CXX = g++

all: innto innblobs innto_hpc innto_hw

innto: innto.cpp dft_dft_solver.h drt_dft_solver.h blueprint.h equations.h
	$(CXX) -O3 $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@

innto_per: innto_per.cpp dft_dft_solver.h blueprint.h equations.h
	$(CXX) -O3 $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@

innblobs: innblobs.cpp dft_dft_solver.h drt_dft_solver.h blueprint.h equations.h
	$(CXX) -O3 $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@

innto_t: innto.cpp dft_dft_solver.h drt_dft_solver.h blueprint.h equations.h
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@

innto_hpc: innto_hpc.cpp dft_dft_solver.h blueprint.h equations.h
	$(CXX) -O3 $< $(CFLAGS) $(INCLUDE) $(LIBS) -lhdf5 -lhdf5_hl -o $@

innto_hw: innto_hw.cpp dft_dft_solver.h blueprint.h equations.h energetics.h
	$(CXX) -O2 $< $(CFLAGS) $(INCLUDE) $(LIBS) -lhdf5 -lhdf5_hl -o $@

energetics_t: energetics_t.cpp energetics.h
	$(CXX) -O2 $< $(CFLAGS) $(INCLUDE) -lm -lfftw3 -o $@

%_t: %_t.cpp %.h
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@

%_b: %_t.cpp %.h
	$(CXX) -O3  $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@
clean:
	rm -f *_t *_b innto innblobs innto_hpc
