INCLUDE = -I../../../
CXX = g++
CFLAGS = -Wall -fopenmp -std=c++0x
LIBS = -lfftw3 -lm
DEBUG = 

GLFLAGS=-lglfw3 -lXxf86vm -lXext -lX11 -lrt -lGLU -lGL -lXi -lXrandr #you might add some libs here, check your glfw installation

toefl: toefl_b.cpp
	$(CXX) $(DEBUG) -O3 $< $(CFLAGS) $(INCLUDE) $(LIBS) $(GLFLAGS) -o $@
	./$@

toefl_t: toefl_t.cpp
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) $(GLFLAGS) -o $@
	./$@

toefl_b: toefl_b.cpp
	$(CXX) $(DEBUG) -O3 $< $(CFLAGS) $(INCLUDE) $(LIBS) $(GLFLAGS) -o $@
	./$@

texture_t: texture_t.cpp texture.h
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) $(GLFLAGS) -o $@
	./$@


%_t: %_t.cpp %.h
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@

%_b: %_b.cpp %.h
	$(CXX) -O3  $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@

.PHONY: doc clean

doc:
	doxygen Doxyfile

clean:
	rm -f *_t *_b toefl
