###
##Packages needed on linux for compilation of doxygen doc
##sudo apt install
#doxygen
#libjs-mathjax (for nice mathematical formulas)
#graphviz (for nice graphical nodes)
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

# needs doxygen >= 1.9.1 (for the sidebar only theme)
dg: exblas.tag geometries.tag
	cd ../inc/dg; \
		(cat Doxyfile; \
		echo "OUTPUT_DIRECTORY = ../../doc/dg"; \
		echo "HTML_HEADER = ../../doc/header.html"; \
		echo "HTML_FOOTER = ../../doc/footer.html"; \
		echo "HTML_EXTRA_STYLESHEET = ../../doc/doxygen-awesome.css";\
		echo "HTML_EXTRA_STYLESHEET += ../../doc/doxygen-awesome-sidebar-only.css";\
		echo "HTML_EXTRA_STYLESHEET += ../../doc/menubar.css";\
    	echo "EXTERNAL_GROUPS=NO" ;\
    	echo "EXTERNAL_PAGES=NO" ;\
		echo "TAGFILES = ../../doc/exblas.tag=../../exblas/html ../../doc/geometries.tag=../../geometries/html") | doxygen - ;

# MW don't know what these are good for (recommended by doxygen-awesome)
#echo "HTML_COLORSTYLE_GAMMA = 113";\
#echo "HTML_COLORSTYLE_HUE = 209";\
#echo "HTML_COLORSTYLE_SAT = 255";\

geometries: dg.tag
	cd ../inc/geometries; \
		(cat Doxyfile; \
		echo "OUTPUT_DIRECTORY = ../../doc/geometries"; \
		echo "HTML_HEADER = ../../doc/header.html"; \
		echo "HTML_FOOTER = ../../doc/footer.html"; \
		echo "HTML_EXTRA_STYLESHEET = ../../doc/doxygen-awesome.css";\
		echo "HTML_EXTRA_STYLESHEET += ../../doc/doxygen-awesome-sidebar-only.css";\
		echo "HTML_EXTRA_STYLESHEET += ../../doc/menubar.css";\
    	echo "EXTERNAL_GROUPS=NO" ;\
    	echo "EXTERNAL_PAGES=NO" ;\
		echo "TAGFILES = ../../doc/dg.tag=../../dg/html") | doxygen - ;

file:
	cd ../inc/file; \
		(cat Doxyfile; \
		echo "OUTPUT_DIRECTORY = ../../doc/file"; \
		echo "HTML_HEADER = ../../doc/header.html"; \
		echo "HTML_FOOTER = ../../doc/footer.html"; \
		echo "HTML_EXTRA_STYLESHEET = ../../doc/doxygen-awesome.css";\
		echo "HTML_EXTRA_STYLESHEET += ../../doc/doxygen-awesome-sidebar-only.css";\
		echo "HTML_EXTRA_STYLESHEET += ../../doc/menubar.css";\
		echo ) | doxygen - ;

exblas:
	cd ../inc/dg/backend/exblas; \
		(cat Doxyfile; \
		echo "OUTPUT_DIRECTORY = ../../../../doc/exblas"; \
		echo "HTML_HEADER = ../../../../doc/header.html"; \
		echo "HTML_FOOTER = ../../../../doc/footer.html"; \
		echo "GENERATE_TREEVIEW = YES"; \
		echo "HTML_EXTRA_STYLESHEET = ../../../../doc/doxygen-awesome.css";\
		echo "HTML_EXTRA_STYLESHEET += ../../../../doc/doxygen-awesome-sidebar-only.css";\
		echo "HTML_EXTRA_STYLESHEET += ../../../../doc/menubar.css";\
		echo ) | doxygen - ;


doc: dg geometries file exblas
	ln -sf dg/html/modules.html index.html
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
	#Open with:     firefox index.html                                        #
	#or on Windows: firefox dg/html/modules.html                              #
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

full_doc: doc

clean:
	rm -rf dg file geometries exblas dg.tag file.tag geometries.tag exblas.tag index.html
