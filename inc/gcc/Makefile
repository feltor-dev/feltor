
INCLUDE += 
CXX = g++ 
CFLAGS = -Wall -std=c++0x
LIBS = -lfftw3 -lm
DEBUG = 

%_t: %_t.cpp %.h
	$(CXX) -DDG_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@
%_b: %_b.cpp %.h
	$(CXX) -O3  $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@


.PHONY: clean

clean:
	rm -f *_t *_b
