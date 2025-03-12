###
##Packages needed on linux for compilation of doxygen doc
##sudo apt install

#doxygen >= 1.10 <= 1.12
#libjs-mathjax (for nice mathematical formulas)
#graphviz (for nice graphical nodes)
#doxygen-awesome

# Optionally clone https://github.com/jothepro/doxygen-awesome-css (current version 2.3.1)
# and use make install
#
# We followed doxygen-awesome documentation and use menubar.css for some custom
# code colors and header.html with some additions to make the menubar and
# activate doxygen-awesome extensions

# The online documentation is hosted on the https://github.com/mwiesenberger/feltor fork
# To update the online documentation we use ghp-import with the -o option, such
# that the gh-pages always
# contains only a single commit (to save space; and the history of documentation
# is contained in the main branch anyway).
#```bash
#pip install ghp-import
#cd path/to/feltor/doc
#make clean
#make doc
#cd ..
#ghp-import -n -f -p -o doc
# # or
#ghp-import -n -f -p -o --remote=mygithub doc
#```

###

# doxygen-awesome-css path
#  if installed via system package manager
#dapath=/usr/share/doxygen-awesome-css
# installed via make install
dapath=/usr/local/share/doxygen-awesome-css
# Relative paths won't work!!
#dapath=../../doxygen-awesome-css

all: doc

.PHONY: clean doc exblas.tag dg.tag geometries.tag matrix.tag dg file geometries exblas matrix

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

matrix.tag:
	cd ../inc/matrix;  \
		(cat Doxyfile; \
		echo "OUTPUT_DIRECTORY = ../../doc/matrix"; \
		echo "GENERATE_HTML = NO"; \
		echo "GENERATE_TAGFILE = ../../doc/matrix.tag" ) | doxygen - ;


dg: exblas.tag geometries.tag matrix.tag
	cd ../inc/dg; \
		(cat Doxyfile; \
		echo "OUTPUT_DIRECTORY = ../../doc/dg"; \
		echo "HTML_HEADER = ../../doc/header.html"; \
		echo "HTML_FOOTER = ../../doc/footer.html"; \
		echo "HTML_EXTRA_STYLESHEET = $(dapath)/doxygen-awesome.css";\
		echo "HTML_EXTRA_STYLESHEET += $(dapath)/doxygen-awesome-sidebar-only.css";\
		echo "HTML_EXTRA_STYLESHEET += $(dapath)/doxygen-awesome-fragment-copy-button.js";\
		echo "HTML_EXTRA_STYLESHEET += ../../doc/menubar.css";\
		echo "TAGFILES = ../../doc/exblas.tag=../../exblas/html ../../doc/geometries.tag=../../geometries/html" ../../doc/matrix.tag=../../matrix/html) | doxygen - ;

geometries: dg.tag
	cd ../inc/geometries; \
		(cat Doxyfile; \
		echo "OUTPUT_DIRECTORY = ../../doc/geometries"; \
		echo "HTML_HEADER = ../../doc/header.html"; \
		echo "HTML_FOOTER = ../../doc/footer.html"; \
		echo "HTML_EXTRA_STYLESHEET = $(dapath)/doxygen-awesome.css";\
		echo "HTML_EXTRA_STYLESHEET += $(dapath)/doxygen-awesome-sidebar-only.css";\
		echo "HTML_EXTRA_STYLESHEET += $(dapath)/doxygen-awesome-fragment-copy-button.js";\
		echo "HTML_EXTRA_STYLESHEET += ../../doc/menubar.css";\
		echo "TAGFILES = ../../doc/dg.tag=../../dg/html") | doxygen - ;

matrix: dg.tag
	cd ../inc/matrix; \
		(cat Doxyfile; \
		echo "OUTPUT_DIRECTORY = ../../doc/matrix"; \
		echo "HTML_HEADER = ../../doc/header.html"; \
		echo "HTML_FOOTER = ../../doc/footer.html"; \
		echo "HTML_EXTRA_STYLESHEET = $(dapath)/doxygen-awesome.css";\
		echo "HTML_EXTRA_STYLESHEET += $(dapath)/doxygen-awesome-sidebar-only.css";\
		echo "HTML_EXTRA_STYLESHEET += $(dapath)/doxygen-awesome-fragment-copy-button.js";\
		echo "HTML_EXTRA_STYLESHEET += ../../doc/menubar.css";\
		echo "TAGFILES = ../../doc/dg.tag=../../dg/html") | doxygen - ;

file:
	cd ../inc/file; \
		(cat Doxyfile; \
		echo "OUTPUT_DIRECTORY = ../../doc/file"; \
		echo "HTML_HEADER = ../../doc/header.html"; \
		echo "HTML_FOOTER = ../../doc/footer.html"; \
		echo "HTML_EXTRA_STYLESHEET = $(dapath)/doxygen-awesome.css";\
		echo "HTML_EXTRA_STYLESHEET += $(dapath)/doxygen-awesome-sidebar-only.css";\
		echo "HTML_EXTRA_STYLESHEET += $(dapath)/doxygen-awesome-fragment-copy-button.js";\
		echo "HTML_EXTRA_STYLESHEET += ../../doc/menubar.css";\
		echo ) | doxygen - ;

exblas:
	cd ../inc/dg/backend/exblas; \
		(cat Doxyfile; \
		echo "OUTPUT_DIRECTORY = ../../../../doc/exblas"; \
		echo "HTML_HEADER = ../../../../doc/header.html"; \
		echo "HTML_FOOTER = ../../../../doc/footer.html"; \
		echo "HTML_EXTRA_STYLESHEET = $(dapath)/doxygen-awesome.css";\
		echo "HTML_EXTRA_STYLESHEET += $(dapath)/doxygen-awesome-sidebar-only.css";\
		echo "HTML_EXTRA_STYLESHEET += $(dapath)/doxygen-awesome-fragment-copy-button.js";\
		echo "HTML_EXTRA_STYLESHEET += ../../../../doc/menubar.css";\
		echo ) | doxygen - ;


doc: dg geometries file exblas matrix
	ln -sf dg/html/topics.html index.html
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
	#Open with:     firefox index.html                                        #
	#or on Windows: firefox dg/html/topics.html                              #
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

full_doc: doc

clean:
	rm -rf dg file geometries matrix exblas dg.tag file.tag geometries.tag matrix.tag exblas.tag index.html
