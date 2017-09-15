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
	echo "MATHJAX_RELPATH = http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML";  \
	echo "GENERATE_TAGFILE = ./dg.tag" ) | doxygen - ;


file:
	(cat ../file/Doxyfile; \
    mkdir -p file; \
	echo "INPUT = ../file/"; \
	echo "OUTPUT_DIRECTORY = ./file"; \
	echo "HTML_HEADER = header.html"; \
	echo "MATHJAX_RELPATH = http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"; \
	echo "GENERATE_TAGFILE = ./file.tag" ) | doxygen - ; 

geometries:
	(cat ../geometries/Doxyfile; \
	echo "INPUT = ../geometries/"; \
    echo "OUTPUT_DIRECTORY = ./geometries/";  \
	echo "HTML_HEADER = header.html"; \
    echo "EXTERNAL_GROUPS=NO" ;\
    echo "EXTERNAL_PAGES=NO" ;\
	echo "MATHJAX_RELPATH = http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML";  \
    echo "TAGFILES = $(tagfiles)") | doxygen - ; 

doc: dg file geometries
	ln -sf dg/html/modules.html index.html
	#Open with: firefox index.html or on Windows: firefox dg/html/modules.html


clean:
	rm -rf dg file geometries dg.tag file.tag index.html


