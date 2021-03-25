device=gpu

#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../         # other src libraries
INCLUDE+= -I../../inc   # other project libraries

all: feltor feltor_hpc feltor_mpi manufactured feltordiag interpolate_in_3d

manufactured: manufactured.cu manufactured.h feltor.h implicit.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(JSONLIB)

implicit_t: implicit_t.cu implicit.h  feltor.h implicit.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(JSONLIB) -g -DDG_BENCHMARK

feltordiag: feltordiag.cu feltordiag.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g
interpolate_in_3d: interpolate_in_3d.cu feltordiag.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -g

feltor: feltor.cu feltor.h implicit.h init.h parameters.h init_from_file.h
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(GLFLAGS) $(JSONLIB) -g -DDG_BENCHMARK

feltor_hpc: feltor.cu feltordiag.h feltor.h implicit.h init.h parameters.h init_from_file.h
	$(CC) -g $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DDG_BENCHMARK -DWITHOUT_GLFW

feltor_mpi: feltor.cu feltordiag.h feltor.h implicit.h init.h parameters.h init_from_file.h
	$(MPICC) $(OPT) $(MPICFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) -DWITH_MPI -DDG_BENCHMARK -DWITHOUT_GLFW

.PHONY: clean doc

doc:
	pdflatex -shell-escape feltor.tex
	bibtex feltor.aux;
	pdflatex -shell-escape feltor.tex
	pdflatex -shell-escape feltor.tex

clean:
	rm -f feltor feltor_hpc feltor_mpi manufactured feltordiag interpolate_in_3d
