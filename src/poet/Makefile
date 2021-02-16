device=omp

#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: poet poet_hpc

poet: poet.cu poet.cuh
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(JSONLIB) -DDG_BENCHMARK  -g
	
poet_hpc: poet_hpc.cu poet.cuh
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DDG_BENCHMARK  -g

poet_mpi: poet_hpc.cu poet.cuh
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DTOEFL_MPI -DDG_BENCHMARK -g

doc:
	mkdir -p doc;
	pdflatex -output-directory doc ./poet.tex;
	bibtex doc/poet.aux;
	pdflatex -output-directory doc ./poet.tex;
	pdflatex -output-directory doc ./poet.tex;


.PHONY: clean doc

clean:
	rm -rf poet poet_hpc poet_mpi doc
