INCLUDE += -I../../inc
CFLAGS = -Wall -std=c++0x -fopenmp
LIBS = -lfftw3 -lm -lGL -lglfw -lrt -lX11 -lXxf86vm
CXX = g++

all: innto innblobs innto_hpc

innto: innto.cpp dft_dft_solver.h drt_dft_solver.h blueprint.h equations.h
	$(CXX) -O3 $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@

innblobs: innblobs.cpp dft_dft_solver.h drt_dft_solver.h blueprint.h equations.h
	$(CXX) -O3 $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@

innto_t: innto.cpp dft_dft_solver.h drt_dft_solver.h blueprint.h equations.h
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@

innto_hpc: innto_hpc.cpp drt_dft_solver.h blueprint.h equations.h
	$(CXX) -O3 $< $(CFLAGS) $(INCLUDE) $(LIBS) -lhdf5 -lhdf5_hl -o $@


%_t: %_t.cpp %.h
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@

%_b: %_t.cpp %.h
	$(CXX) -O3  $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@
clean:
	rm -f *_t *_b innto innblobs
