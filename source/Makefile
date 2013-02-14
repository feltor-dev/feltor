INCLUDE += -I../lib
CFLAGS = -Wall -std=c++0x -fopenmp
LIBS = -lfftw3 -lm -lGL -lglfw -lrt
CXX = g++

all: innto

innto: innto.cpp dft_dft_solver.h drt_dft_solver.h blueprint.h equations.h
	$(CXX) -O3 $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@

innto_t: innto.cpp dft_dft_solver.h drt_dft_solver.h blueprint.h equations.h
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@

%_t: %_t.cpp %.h
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@

%_b: %_t.cpp %.h
	$(CXX) -O3  $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@
clean:
	rm *_t *_b innto 
