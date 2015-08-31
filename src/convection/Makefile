INCLUDE = -I../../inc
INCLUDE += -I$(HOME)/include

CFLAGS = -Wall -std=c++0x -fopenmp
LIBS = $$(pkg-config --static --libs glfw3) -lfftw3
CXX = g++

all: convection

convection: convection.cpp convection_solver.h 
	$(CXX) -O3 $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@

clean:
	rm -f convection 
