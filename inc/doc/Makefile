###Packages needed on linux:
###doxygen
###libjs-mathjax
###graphviz
#(shouldn't require latex according to doxygen documentation)

all: doc

.PHONY: clean doc dg.tag dg file geometries

dg.tag:
	(cat ../dg/Doxyfile; \
	echo "INPUT = ../dg/"; \
	echo "OUTPUT_DIRECTORY = ./dg"; \
	echo "GENERATE_HTML = NO"; \
	echo "GENERATE_TAGFILE = ./dg.tag" ) | doxygen - ;

dg:
	(cat ../dg/Doxyfile; \
	echo "INPUT = ../dg/"; \
	echo "OUTPUT_DIRECTORY = ./dg"; \
	echo "HTML_HEADER = header.html"; \
    echo "EXTERNAL_GROUPS=NO" ;\
    echo "EXTERNAL_PAGES=NO" ;\
	echo ) | doxygen - ; 

geometries: dg.tag
	(cat ../geometries/Doxyfile; \
	echo "INPUT = ../geometries/"; \
	echo "OUTPUT_DIRECTORY = ./geometries"; \
	echo "HTML_HEADER = header.html"; \
    echo "EXTERNAL_GROUPS=NO" ;\
    echo "EXTERNAL_PAGES=NO" ;\
	echo "TAGFILES = dg.tag=../../dg/html") | doxygen - ; 

file:
	(cat ../file/Doxyfile; \
	echo "INPUT = ../file/"; \
	echo "OUTPUT_DIRECTORY = ./file"; \
	echo "HTML_HEADER = header.html"; \
	echo ) | doxygen - ; 

doc: dg geometries file
	ln -sf dg/html/modules.html index.html
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
	#Open with: firefox index.html or on Windows: firefox dg/html/modules.html#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

clean:
	rm -rf dg file geometries dg.tag file.tag geometries.tag index.html
