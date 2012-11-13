INCLUDE += 
CFLAGS = -Wall 
LIBS = -lfftw3 -lm
CXX = g++

all: matrix_t fft_t arakawa_t coeff_t quadmat_t dft_dst_t

%_t: %_t.cpp %.h
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@

%_b: %_b.cpp %.h
	$(CXX) -O3  $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./$@
clean:
	rm *_t *_b
