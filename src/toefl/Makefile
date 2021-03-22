device=gpu

#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: toeflR toefl_hpc toefl_mpi

toeflR: toeflR.cu toeflR.cuh parameters.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(JSONLIB) -DDG_BENCHMARK  -g

toefl_hpc: toefl_hpc.cu toeflR.cuh parameters.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DDG_BENCHMARK  -g

toefl_mpi: toefl_hpc.cu toeflR.cuh parameters.h
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DWITH_MPI -DDG_BENCHMARK -g

doc:
	pdflatex -shell-escape ./toefl.tex;
	bibtex toefl.aux;
	pdflatex -shell-escape ./toefl.tex;
	pdflatex -shell-escape ./toefl.tex;


.PHONY: clean doc

clean:
	rm -rf toeflR toefl_hpc toefl_mpi toefl.aux toefl.log toefl.out toefl.pyg toefl.pdf toefl.bbl toefl.blg
