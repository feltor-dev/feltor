INCLUDE = -I../../inc
INCLUDE += -I/home/markus/Dokumente/Phd/Include

CFLAGS = -Wall -std=c++0x -fopenmp
LIBS = -lfftw3 -lm -lGL -lglfw3 -lrt -lX11 -lXxf86vm -lXi -lXrandr
CXX = g++

all: convection

convection: convection.cpp convection_solver.h 
	$(CXX) -O3 $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@

clean:
	rm -f convection 
