device=gpu

#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: poet poet_hpc poet_mpi

poet: poet.cpp poet.h init.h diag.h parameters.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(LIBS) $(JSONLIB) $(LAPACKLIB) -DDG_BENCHMARK  -g

poet_hpc: poet.cpp poet.h init.h diag.h parameters.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) $(LAPACKLIB) -DDG_BENCHMARK -DWITHOUT_GLFW

poet_mpi: poet.cpp poet.h init.h diag.h parameters.h
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) $(LAPACKLIB) -DWITH_MPI -DDG_BENCHMARK -DWITHOUT_GLFW

doc:
	pdflatex --shell-escape ./poet.tex;
	bibtex poet.aux;
	pdflatex --shell-escape ./poet.tex;
	pdflatex --shell-escape ./poet.tex;


.PHONY: clean doc

clean:
	rm -rf poet poet_hpc poet_mpi doc
