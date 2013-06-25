CXX=g++


read_input_t: read_input_t.cpp read_input.h 
	$(CXX) $< -o $@ 


.PHONY: doc clean

doc:
	doxygen Doxyfile


clean:
	rm -f read_input_t
