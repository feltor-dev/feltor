INCLUDE = -I../../inc
INCLUDE += -I../spectral

CFLAGS = -Wall -std=c++0x -fopenmp
LIBS = -lfftw3 -lm -lGL -lglfw -lrt -lX11 -lXxf86vm
CXX = g++

all: convection

convection: convection.cpp convection_solver.h 
	$(CXX) -O3 $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@

clean:
	rm -f convection 
