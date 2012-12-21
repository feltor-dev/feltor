INCLUDE += 
CXX = g++
CFLAGS = -Wall -std=c++0x
LIBS = -lfftw3 -lm

GLFLAGS=-lglfw -lrt -lGL

toefl: toefl_b.cpp
	$(CXX) -O3 $< $(CFLAGS) $(INCLUDE) $(LIBS) $(GLFLAGS) -o $@
	./$@

toefl_t: toefl_t.cpp
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) $(GLFLAGS) -o $@
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
clean:
	rm *_t *_b
