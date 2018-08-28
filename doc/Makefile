###
##Packages needed on linux for compilation of both doxygen and latex doc
##sudo apt install
#doxygen
#libjs-mathjax (for nice mathematical formulas)
#graphviz (for nice graphical nodes)
#texlive  (for pdflatex)
#texlive-latex-extra (for the hitec.cls package)
###

all: doc

.PHONY: clean doc exblas.tag dg.tag geometries.tag dg file geometries exblas

# the semicolons and backslashes are needed by Makefile
dg.tag:
	cd ../inc/dg; \
		(cat Doxyfile; \
		echo "OUTPUT_DIRECTORY = ../../doc/dg"; \
		echo "GENERATE_HTML = NO"; \
		echo "GENERATE_TAGFILE = ../../doc/dg.tag" ) | doxygen - ;

exblas.tag:
	cd ../inc/dg/backend/exblas; \
		(cat Doxyfile; \
		echo "OUTPUT_DIRECTORY = ../../../../doc/exblas"; \
		echo "GENERATE_HTML = NO"; \
		echo "GENERATE_TAGFILE = ../../../../doc/exblas.tag" ) | doxygen - ;

geometries.tag:
	cd ../inc/geometries;  \
		(cat Doxyfile; \
		echo "OUTPUT_DIRECTORY = ../../doc/geometries"; \
		echo "GENERATE_HTML = NO"; \
		echo "GENERATE_TAGFILE = ../../doc/geometries.tag" ) | doxygen - ;

dg: exblas.tag geometries.tag dg_introduction.pdf
	cd ../inc/dg; \
		(cat Doxyfile; \
		echo "OUTPUT_DIRECTORY = ../../doc/dg"; \
		echo "HTML_EXTRA_FILES  = ../../doc/related_pages/dg_introduction/dg_introduction.pdf"; \
		echo "HTML_HEADER = ../../doc/header.html"; \
		echo "HTML_EXTRA_STYLESHEET = ../../doc/menubar.css";\
    	echo "EXTERNAL_GROUPS=NO" ;\
    	echo "EXTERNAL_PAGES=NO" ;\
		echo "TAGFILES = ../../doc/exblas.tag=../../exblas/html ../../doc/geometries.tag=../../geometries/html") | doxygen - ;

geometries: dg.tag parallel.pdf hector.pdf
	cd ../inc/geometries; \
		(cat Doxyfile; \
		echo "OUTPUT_DIRECTORY = ../../doc/geometries"; \
		echo "HTML_EXTRA_FILES  = ../../doc/related_pages/parallel/parallel.pdf"; \
		echo "HTML_EXTRA_FILES += ../../doc/related_pages/hector/hector.pdf"; \
		echo "HTML_HEADER = ../../doc/header.html"; \
		echo "HTML_EXTRA_STYLESHEET = ../../doc/menubar.css";\
    	echo "EXTERNAL_GROUPS=NO" ;\
    	echo "EXTERNAL_PAGES=NO" ;\
		echo "TAGFILES = ../../doc/dg.tag=../../dg/html") | doxygen - ;

file:
	cd ../inc/file; \
		(cat Doxyfile; \
		echo "OUTPUT_DIRECTORY = ../../doc/file"; \
		echo "HTML_HEADER = ../../doc/header.html"; \
		echo "HTML_EXTRA_STYLESHEET = ../../doc/menubar.css";\
		echo ) | doxygen - ;

exblas:
	cd ../inc/dg/backend/exblas; \
		(cat Doxyfile; \
		echo "OUTPUT_DIRECTORY = ../../../../doc/exblas"; \
		echo "HTML_HEADER = ../../../../doc/header.html"; \
		echo "HTML_EXTRA_STYLESHEET = ../../../../doc/menubar.css";\
		echo ) | doxygen - ;


%.pdf:
	cd related_pages/$*; \
		pdflatex $*.tex; \
		bibtex $*.aux; \
		pdflatex $*.tex; \
		pdflatex $*.tex;


doc: dg geometries file exblas
	ln -sf dg/html/modules.html index.html
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
	#Open with:     firefox index.html                                        #
	#or on Windows: firefox dg/html/modules.html                              #
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

full_doc: pdf doc

clean:
	rm -rf dg file geometries exblas dg.tag file.tag geometries.tag exblas.tag index.html
