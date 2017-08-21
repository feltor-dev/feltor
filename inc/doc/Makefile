tagfiles="./dg.tag=../../dg/html ./file.tag=../../file/html"
.PHONY: dg file geometries

all: doc

dg:
	(cat ../dg/Doxyfile; \
    mkdir -p dg; \
	echo "INPUT = ../dg/"; \
	echo "OUTPUT_DIRECTORY = ./dg"; \
	echo "HTML_HEADER = header.html"; \
	echo "GENERATE_TAGFILE = ./dg.tag" ) | doxygen - ;
	#echo "MATHJAX_RELPATH = http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"; \ ## uncomment if run for official feltor page


file:
	(cat ../file/Doxyfile; \
    mkdir -p file; \
	echo "INPUT = ../file/"; \
	echo "OUTPUT_DIRECTORY = ./file"; \
	echo "HTML_HEADER = header.html"; \
	echo "GENERATE_TAGFILE = ./file.tag" ) | doxygen - ;
	#echo "MATHJAX_RELPATH = http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"; \ ## uncomment if run for official feltor page

geometries:
	(cat ../geometries/Doxyfile; \
	echo "INPUT = ../geometries/"; \
    echo "OUTPUT_DIRECTORY = ./geometries/";  \
	echo "HTML_HEADER = header.html"; \
    echo "EXTERNAL_GROUPS=NO" ;\
    echo "EXTERNAL_PAGES=NO" ;\
    echo "TAGFILES = $(tagfiles)") | doxygen - ;
	#echo "MATHJAX_RELPATH = http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"; \ ## uncomment if run for official feltor page

doc: dg file geometries
	ln -fs dg/html/modules.html index.html
	echo "Open with: firefox index.html"


clean:
	rm -rf dg file geometries dg.tag file.tag index.html


