
toefl: source/t3l.C
	g++ source/t3l.C -O2 -lfftw3 -lm -fopenmp -lpthread -lfftw3_threads -o t3l

debug: source/t3l.C
	g++ source/t3l.C -lfftw3 -lm -lpthread -lfftw3_threads -o t3l -Wall -g
