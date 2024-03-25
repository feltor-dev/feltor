device=gpu

#configure machine
include ../../config/default.mk
include ../../config/version.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: feltor feltor_hpc feltor_mpi feltordiag interpolate_in_3d

feltordiag: feltordiag.cpp feltordiag.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g $(VERSION_FLAGS)

interpolate_in_3d: interpolate_in_3d.cpp feltordiag.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g $(VERSION_FLAGS)

feltor: feltor.cpp feltordiag.h feltor.h init.h parameters.h init_from_file.h common.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(LIBS) $(JSONLIB) -g $(VERSION_FLAGS)

feltor_hpc: feltor.cpp feltordiag.h feltor.h init.h parameters.h init_from_file.h common.h
	$(CC) -g $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DWITHOUT_GLFW $(VERSION_FLAGS)

feltor_mpi: feltor.cpp feltordiag.h feltor.h init.h parameters.h init_from_file.h common.h
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DWITH_MPI -DWITHOUT_GLFW $(VERSION_FLAGS)

feltor_pol: feltor.cpp feltordiag.h feltor.h init.h parameters.h init_from_file.h common.h
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DWITH_MPI -DWRITE_POL_FILE -DWITHOUT_GLFW $(VERSION_FLAGS)

multigrid_b: multigrid_b.cpp parameters.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g $(VERSION_FLAGS)

convert_b: convert_b.cpp parameters.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g $(VERSION_FLAGS)

.PHONY: clean doc

doc:
	pdflatex -shell-escape feltor.tex
	bibtex feltor.aux;
	pdflatex -shell-escape feltor.tex
	pdflatex -shell-escape feltor.tex

clean:
	rm -f feltor feltor_hpc feltor_mpi feltordiag interpolate_in_3d feltor.bbl feltor.blg feltor.aux feltor.log feltor.out feltor.toc
