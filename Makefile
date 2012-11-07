.PHONY: doc
doc: 
	doxygen Doxyfile

%.pdf: %.tex
	latex $*
	dvips $*
	ps2pdf $*.ps
	rm $*.dvi $*.log $*.ps 

