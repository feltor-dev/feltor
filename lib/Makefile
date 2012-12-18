INCLUDE += 
CFLAGS = -Wall -std=c++0x
LIBS = -lfftw3 -lm
CXX = g++

toefl_t: toefl_t.cpp
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) -lglfw -lGL -o $@
	./$@

texture_t: texture_t.cpp texture.h
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) -lglfw -lGL -o $@
	./$@

toefl: toefl_b.cpp
	$(CXX) -O3 $< $(CFLAGS) $(INCLUDE) $(LIBS) -lglfw -lGL -o $@
	./$@
    

%_t: %_t.cpp %.h
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@

%_b: %_b.cpp %.h
	$(CXX) -O3  $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@
clean:
	rm *_t *_b
