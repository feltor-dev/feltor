device=gpu

#configure machine
include ../../config/default.mk
include ../../config/version.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: reconnection reconnection_hpc reconnection_mpi

reconnection: reconnection.cpp reconnection.h init.h diag.h parameters.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(LIBS) $(JSONLIB) $(VERSION_FLAGS) -g

reconnection_hpc: reconnection.cpp reconnection.h init.h diag.h parameters.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) $(VERSION_FLAGS) -DWITHOUT_GLFW

reconnection_mpi: reconnection.cpp reconnection.h init.h diag.h parameters.h
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) $(VERSION_FLAGS) -DWITH_MPI -DWITHOUT_GLFW

.PHONY: clean

doc:
	pdflatex -shell-escape ./reconnection.tex;
	bibtex reconnection.aux;
	pdflatex -shell-escape ./reconnection.tex;
	pdflatex -shell-escape ./reconnection.tex;

clean:
	rm -f reconnection reconnection_hpc reconnection_mpi reconnection.aux reconnection.bbl reconnection.blg reconnection.log reconnection.out reconnection.toc
