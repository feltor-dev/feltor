tagfiles="./dg.tag=../../dg/html ./file.tag=../../file/html"
.PHONY: dg file geometries

###Packages needed on linux:
###doxygen
###libjs-mathjax
###graphviz
#(shouldn't require latex according to doxygen documentation)

all: doc

dg:
	(cat ../dg/Doxyfile; \
    mkdir -p dg; \
	echo "INPUT = ../dg/"; \
	echo "OUTPUT_DIRECTORY = ./dg"; \
	echo "HTML_HEADER = header.html"; \
	echo "GENERATE_TAGFILE = ./dg.tag" ) | doxygen - ;\
	echo "MATHJAX_RELPATH = http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"; 


file:
	(cat ../file/Doxyfile; \
    mkdir -p file; \
	echo "INPUT = ../file/"; \
	echo "OUTPUT_DIRECTORY = ./file"; \
	echo "HTML_HEADER = header.html"; \
	echo "GENERATE_TAGFILE = ./file.tag" ) | doxygen - ; \
	echo "MATHJAX_RELPATH = http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"; 

geometries:
	(cat ../geometries/Doxyfile; \
	echo "INPUT = ../geometries/"; \
    echo "OUTPUT_DIRECTORY = ./geometries/";  \
	echo "HTML_HEADER = header.html"; \
    echo "EXTERNAL_GROUPS=NO" ;\
    echo "EXTERNAL_PAGES=NO" ;\
    echo "TAGFILES = $(tagfiles)") | doxygen - ; \
	echo "MATHJAX_RELPATH = http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"; 

doc: dg file geometries
	ln -sf dg/html/modules.html index.html
	echo "Open with: firefox index.html or on Windows: firefox dg/html/modules.html"


clean:
	rm -rf dg file geometries dg.tag file.tag index.html


