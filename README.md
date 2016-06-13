<hr>
<h3> Welcome to the FELTOR project!</h3>

For the CPU-Version you'll need to download [thrust]( https://github.com/thrust/thrust/tree/1.8.1) (v1.8.1)

and [cusp] (https://github.com/cusplibrary/cusplibrary/tree/0.5.0)
( v0.5.0 )

If you want to compile for a GPU you need to install [CUDA](https://developer.nvidia.com/cuda-zone) (minimum v7.0 recommended)
However, we explicitely note that you need CUDA and a GPU only if you 
want to actually use them. It's possible to use FELTOR completely without these.

For the mpi version you need an mpi compiler. 

Feltor compiles fine with gcc-4.8

If you want to have a nice OpenGL window while computing download the [draw library]( https://github.com/mwiesenberger/draw)
which depends on OpenGL (s.a. [installation guide] (http://en.wikibooks.org/wiki/OpenGL_Programming) and [glfw] (http://www.glfw.org) 

If you want to write to the disk install [netcdf4] (http://www.unidata.ucar.edu/software/netcdf/docs/index.html), which is based on hdf5.
If you intend to use the mpi parallelized programs you need to build
hdf5 and netcdf with the --enable-parallel flag. Do NOT use the pnetcdf library, which
uses the classic netcdf file format. 

We start to use JSON as a file format for input files in our application.
You will want to download and install the open-source parser jsoncpp (https://github.com/open-source-parsers/jsoncpp) version 0.10.6

The documentation can be built with doxygen. However, Doxygen itself is not part 
of our library. 


If you want to compile applications in FELTOR you will want to check the 
config folder. Here, machine specific Makefile variables are defined. 
The default.mk file gives an example. You can use the default 
if you create an include folder in your home directory and create 
links to the aforementioned libraries. Or you can 
create your own file. 
