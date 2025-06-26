device=gpu

#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: esol esol_hpc esol_mpi

esol: esol.cpp esol.h init.h diag.h parameters.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(LIBS) $(JSONLIB) $(LAPACKLIB) -g

esol_hpc: esol.cpp esol.h init.h diag.h parameters.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) $(LAPACKLIB) -DWITHOUT_GLFW

esol_mpi: esol.cpp esol.h init.h diag.h parameters.h
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) $(LAPACKLIB) -DWITH_MPI -DWITHOUT_GLFW

doc:
	pdflatex -shell-escape esol.tex;
	bibtex esol.aux;
	pdflatex -shell-escape esol.tex;
	pdflatex -shell-escape esol.tex;


.PHONY: clean doc

clean:
	rm -rf esol esol_hpc esol_mpi doc
