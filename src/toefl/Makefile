device=gpu

#configure machine
include ../../config/default.mk
include ../../config/version.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../../inc   # other project libraries

all: toefl toefl_hpc toefl_mpi

toefl: toefl.cpp toefl.h parameters.h diag.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(LIBS) $(JSONLIB) $(VERSION_FLAGS) -DWITH_GLFW -g

toefl_hpc: toefl.cpp toefl.h parameters.h diag.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) $(VERSION_FLAGS) -g

toefl_mpi: toefl.cpp toefl.h parameters.h diag.h
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) $(VERSION_FLAGS)

doc:
	pdflatex -shell-escape ./toefl.tex;
	bibtex toefl.aux;
	pdflatex -shell-escape ./toefl.tex;
	pdflatex -shell-escape ./toefl.tex;


.PHONY: clean doc

clean:
	rm -rf toefl toefl_hpc toefl_mpi toefl.aux toefl.log toefl.out toefl.pyg toefl.pdf toefl.bbl toefl.blg
