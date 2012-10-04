#gnuplot test
reset

# Ausgabeformat:
set term x11
#set term postscript
#set term png

#set output "ygauss-nz.ps"
#set size 1.0,1.0
#set xlabel 'x'
#set ylabel 'y'
##unset colorbox
#set title " "


#filename = "p2d.dat"
#filename = "n2d.dat"
#filename = "d2d.dat"
filename = "w2d.dat"
#filename = "r2d.dat"
#filename = "zfx.dat"
#filename = "pol2d.dat"


# einfacher 2D Farbplot
##
#set xrange [0 : 511] 
#set yrange [0 : 511] 
#set zrange [0. : 1.4] 
#set cbrange [0. : 1.2] 
#set size square

#set palette defined (-2 "blue", -1 "cyan", 0 "yellow", 1 "orange", 2 "red")
#set palette defined (-1 "blue", 0 "white", 1 "red")
set palette defined (-1.25 "black", -1 "blue", 0 "white", 1 "red", 1.25 "black")
#set palette defined (-1 "black", 0 "red", 1 "yellow")
#set palette rgbformulae 22,13,-31
#
#set palette model RGB
#set palette defined
#
set pm3d map
splot filename notitle
#splot filename every 4 notitle


# 2D Farbplot mit Konturen
#
#set contour base; set cntrparam level 1
#unset surface
#set table 'contour.dat'
#splot filename 
#unset table
#!awk "NF<2{printf\"\n\"}{print}" <contour.dat >contour1.dat
#reset
#
#set ticslevel 0
#set palette defined (-1 "blue", 0 "white", 1 "red")
#set xrange [0 : 127] 
#set yrange [0 : 127] 
#set size square
#set pm3d map
#splot filename with pm3d, 'contour1.dat' with line lt -1
#!rm contour.dat contour1.dat


pause 1

replot
reread

