device=gpu

#configure machine
include ../../config/default.mk
include ../../config/version.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: feltor feltor_hpc feltor_mpi manufactured manufactured_perp manufactured_para feltordiag interpolate_in_3d

feltordiag: feltordiag.cu feltordiag.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g $(VERSION_FLAGS)

interpolate_in_3d: interpolate_in_3d.cu feltordiag.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g $(VERSION_FLAGS)

feltor: feltor.cu feltor.h init.h parameters.h init_from_file.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(LIBS) $(JSONLIB) -g $(VERSION_FLAGS)

feltor_hpc: feltor.cu feltordiag.h feltor.h init.h parameters.h init_from_file.h
	$(CC) -g $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DWITHOUT_GLFW $(VERSION_FLAGS)

feltor_mpi: feltor.cu feltordiag.h feltor.h init.h parameters.h init_from_file.h
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DWITH_MPI -DWITHOUT_GLFW $(VERSION_FLAGS)

feltor_pol: feltor.cu feltordiag.h feltor.h init.h parameters.h init_from_file.h
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DWITH_MPI -DWRITE_POL_FILE -DWITHOUT_GLFW $(VERSION_FLAGS)

manufactured: manufactured.cu manufactured.h feltor.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(JSONLIB) -DFELTORPARALLEL=1 -DFELTORPERP=1

manufactured_perp: manufactured.cu manufactured.h feltor.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(JSONLIB) -DFELTORPARALLEL=0 -DFELTORPERP=1

manufactured_para: manufactured.cu manufactured.h feltor.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(JSONLIB) -DFELTORPARALLEL=1 -DFELTORPERP=0

.PHONY: clean doc

doc:
	pdflatex -shell-escape feltor.tex
	bibtex feltor.aux;
	pdflatex -shell-escape feltor.tex
	pdflatex -shell-escape feltor.tex

clean:
	rm -f feltor feltor_hpc feltor_mpi manufactured manufactured_perp manufactured_para feltordiag interpolate_in_3d
