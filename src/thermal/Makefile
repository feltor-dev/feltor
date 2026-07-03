device=gpu

#configure machine
include ../../config/default.mk
include ../../config/version.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: thermal thermal_mpi thermaldiag

thermaldiag: thermaldiag.cpp thermaldiag.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g $(VERSION_FLAGS)

# There is no GLFW version
thermal: thermal.cpp thermaldiag.h thermal.h init.h parameters.h init_from_file.h perpendicular.h parallel.h solvers.h collisions.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LAPACKLIB) $(LIBS) $(JSONLIB) -g $(VERSION_FLAGS)

thermal_hpc: thermal
	cp thermal thermal_hpc

thermal_mpi: thermal.cpp thermaldiag.h thermal.h init.h parameters.h init_from_file.h perpendicular.h parallel.h solvers.h collisions.h
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LAPACKLIB) $(LIBS) $(JSONLIB) -DWITH_MPI $(VERSION_FLAGS)

.PHONY: clean doc

doc:
	pdflatex -shell-escape thermal.tex
	bibtex thermal.aux;
	pdflatex -shell-escape thermal.tex
	pdflatex -shell-escape thermal.tex

clean:
	rm -f thermal thermal_hpc thermal_mpi thermaldiag thermal.bbl thermal.blg thermal.aux thermal.log thermal.out thermal.toc
