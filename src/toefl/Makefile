device=gpu

#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: toeflR toefl_hpc

toeflR: toeflR.cu toeflR.cuh
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(JSONLIB) -DDG_BENCHMARK  -g

toefl_hpc: toefl_hpc.cu toeflR.cuh
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DDG_BENCHMARK  -g

toefl_mpi: toefl_hpc.cu toeflR.cuh
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DTOEFL_MPI -DDG_BENCHMARK -g

doc:
	mkdir -p doc;
	pdflatex -output-directory doc ./toefl.tex;
	bibtex doc/toefl.aux;
	pdflatex -output-directory doc ./toefl.tex;
	pdflatex -output-directory doc ./toefl.tex;


.PHONY: clean doc

clean:
	rm -rf toeflR toefl_hpc toefl_mpi doc
