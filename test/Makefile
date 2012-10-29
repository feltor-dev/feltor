INCLUDE += -I/home/matthias/include
CFLAGS = -Wall -std=c++0x
LIBS = -lfftw3
CXX = g++

all: matrix_t fft_t arakawa_t coeff_t quadmat_t

matrix_t: matrix_t.o 
	$(CXX) $+ $(LDFLAGS) $(LIBS) -o $@ 
	./matrix_t


matrix_t.o: matrix_t.cpp
	$(CXX) -c -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) 

fft_t: fft_t.cpp fft.h
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./fft_t

arakawa_t: arakawa_t.cpp arakawa.h
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./arakawa_t

coeff_t: coeff_t.cpp coeff.h
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./coeff_t
quadmat_t: quadmat_t.cpp quadmat.h
	$(CXX) -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) $(LIBS) -o $@
	./quadmat_t

matrix.o: matrix.cpp
	$(CXX) -c -DTL_DEBUG $< $(CFLAGS) $(INCLUDE) 


clean:
	rm *.o
