INCLUDE += -I../lib
CFLAGS = -Wall -std=c++0x
LIBS = -lfftw3 -lm
CXX = g++

all: 

%_t: %_t.cpp %.h
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@

%_b: %_b.cpp %.h
	$(CXX) -O3  $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@
clean:
	rm *_t *_b
