device=cpu

#configure machine
include ../../config/default.mk
include ../../config/version.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../../inc   # other project libraries

all: geometry_diag

geometry_diag: geometry_diag.cpp
	$(CC) $(OPT) $(CFLAGS) $< -o $@ $(INCLUDE) $(LIBS) $(JSONLIB) $(VERSION_FLAGS) -g

doc:
	pdflatex -shell-escape ./geometry_diag.tex;
	bibtex geometry_diag.aux;
	pdflatex -shell-escape ./geometry_diag.tex;
	pdflatex -shell-escape ./geometry_diag.tex;


.PHONY: clean

clean:
	rm -f geometry_diag geometry_diag.aux geometry_diag.log geometry_diag.out geometry_diag.pyg geometry_diag.pdf geometry_diag.bbl geometry_diag.blg
