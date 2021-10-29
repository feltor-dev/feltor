device=gpu

#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: esol esol_hpc esol_mpi

esol: esol.cu esol.h init.h diag.h parameters.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(LIBS) $(JSONLIB) -DDG_BENCHMARK  -g
	
esol_hpc: esol.cu esol.h init.h diag.h parameters.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DDG_BENCHMARK -DWITHOUT_GLFW

esol_mpi: esol.cu esol.h init.h diag.h parameters.h
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DWITH_MPI -DDG_BENCHMARK -DWITHOUT_GLFW

doc:
	pdflatex -shell-escape esol.tex;
	bibtex esol.aux;
	pdflatex -shell-escape esol.tex;
	pdflatex -shell-escape esol.tex;


.PHONY: clean doc

clean:
	rm -rf esol esol_hpc esol_mpi doc
