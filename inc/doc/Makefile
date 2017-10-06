###Packages needed on linux:
###doxygen
###libjs-mathjax
###graphviz
#(shouldn't require latex according to doxygen documentation)

all: doc

.PHONY: clean doc dg.tag geometries.tag dg file geometries

# the semicolons and backslashes are needed by Makefile
dg.tag:
	cd ../dg; \
		(cat Doxyfile; \
		echo "OUTPUT_DIRECTORY = ../doc/dg"; \
		echo "GENERATE_HTML = NO"; \
		echo "GENERATE_TAGFILE = ../doc/dg.tag" ) | doxygen - ;

geometries.tag:
	cd ../geometries;  \
		(cat Doxyfile; \
		echo "OUTPUT_DIRECTORY = ../doc/geometries"; \
		echo "GENERATE_HTML = NO"; \
		echo "GENERATE_TAGFILE = ../doc/geometries.tag" ) | doxygen - ;

dg: geometries.tag
	cd ../dg; \
		(cat Doxyfile; \
		echo "OUTPUT_DIRECTORY = ../doc/dg"; \
		echo "HTML_HEADER = ../doc/header.html"; \
    	echo "EXTERNAL_GROUPS=NO" ;\
    	echo "EXTERNAL_PAGES=NO" ;\
		echo "TAGFILES = ../doc/geometries.tag=../../geometries/html") | doxygen - ; 

geometries: dg.tag
	cd ../geometries; \
		(cat Doxyfile; \
		echo "OUTPUT_DIRECTORY = ../doc/geometries"; \
		echo "HTML_HEADER = ../doc/header.html"; \
    	echo "EXTERNAL_GROUPS=NO" ;\
    	echo "EXTERNAL_PAGES=NO" ;\
		echo "TAGFILES = ../doc/dg.tag=../../dg/html") | doxygen - ; 

file:
	cd ../file; \
		(cat Doxyfile; \
		echo "OUTPUT_DIRECTORY = ../doc/file"; \
		echo "HTML_HEADER = ../doc/header.html"; \
		echo ) | doxygen - ; 

pdf:
	cd ../geometries/related_pages; \
		pdflatex parallel.tex;


doc: dg geometries file
	ln -sf dg/html/modules.html index.html
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
	#Open with: firefox index.html or on Windows: firefox dg/html/modules.html#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

full_doc: pdf doc

clean:
	rm -rf dg file geometries dg.tag file.tag geometries.tag index.html
