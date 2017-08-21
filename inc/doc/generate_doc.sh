#!/bin/bash
#this file automatically loops over inc directories and runs doxygen
# on the Doxyfiles and outputs documentation in the current directory
# It generates tag files so that there can be links between inc documentations and uses the html_header feature of doxygen to create the top menu bar

#pay attention on name collisions in documentations through tag files
independent="dg file"
dependent="geometries"
tagfiles=""

#first generate tag files for independen documentations
for i in $independent;
do 
    (cat ../$i/Doxyfile; \
    mkdir -p $i
	echo "INPUT = ../$i/"; \
    #echo "GENERATE_HTML=NO";\
	echo "OUTPUT_DIRECTORY = ./$i"; \
	echo "HTML_HEADER = header.html"; \
	#echo "MATHJAX_RELPATH = http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"; \ ## uncomment if run for official feltor page
    echo "GENERATE_TAGFILE = ./${i}.tag" ) | doxygen - ;
    tagfiles="$tagfiles ./${i}.tag=../../${i}/html"
done;

#generate dependent documentations
for i in $dependent;
do 
    (cat ../$i/Doxyfile; \
	echo "INPUT = ../$i/"; \
    echo "OUTPUT_DIRECTORY = ./$i/";  \
	echo "HTML_HEADER = header.html"; \
    echo "EXTERNAL_GROUPS=NO" ;\
    echo "EXTERNAL_PAGES=NO" ;\
	#echo "MATHJAX_RELPATH = http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"; \ ## uncomment if run for official feltor page
    echo "TAGFILES = $tagfiles") | doxygen - ;
done;

