### Welcome to the FELTOR project!

![3dsimulation](3dpic.jpg)

#### 1. License 
FELTOR is free software and licensed under the very permissive MIT license. It was originally developed by Matthias Wiesenberger and Markus Held.

#### 2. Dependencies 
Our code only depends on external libraries that are themselves openly available. We note here that we do not distribute copies of these libraries. The user has to obtain these according to his/her needs. 

The most basic applications depend on [thrust]( https://github.com/thrust/thrust) and [cusp] (https://github.com/cusplibrary/cusplibrary) distributed under the Apache-2.0 license. The minimum requirement to compile and run an application is a working C++ compiler and a CPU. 
We don't use new C++-11 standard features to avoid complications since some clusters are a bit behind on up-to-date compilers. 
Our GPU backend uses the [Nvidia-CUDA](https://developer.nvidia.com/cuda-zone) programming environment and in order to compile and run the program a user needs the nvcc CUDA compiler (available free of charge) and a NVidia GPU. However, we explicitly note here that due to the modular design of our software a user does not necessarily have to possess a GPU nor the nvcc compiler. The CPU version of the backend is equally valid and provides the same functionality. 
For data output we use the [NetCDF](http://www.unidata.ucar.edu/software/netcdf/) library under an MIT - like license. The [HDF5](https://www.hdfgroup.org/HDF5/) library also uses a very permissive license.  
Data input needs [JsonCpp](https://www.github.com/open-source-parsers/jsoncpp) distributed under the MIT license (the 0.y.x branch to avoid C++-11 support).
Some rendering applications in FELTOR use the [draw library]( https://github.com/mwiesenberger/draw) (developed by us also under MIT), 
which depends on OpenGL (s.a. [installation guide](http://en.wikibooks.org/wiki/OpenGL_Programming)) and [glfw] (http://www.glfw.org), an OpenGL development library under a BSD-like license. The documentation can be generated with [Doxygen](http://www.doxygen.org), however, Doxygen itself is not part of FELTOR. 
If the user intends to use the MPI backend, an implementation library of the mpi standard is required. Note that for the mpi versions of applications you need to build
hdf5 and netcdf with the --enable-parallel flag. Do NOT use the pnetcdf library, which
uses the classic netcdf file format.  The OpenMP standard is natively supported by most recent C++ compilers.

#### 3. Compilation 
To simplify the compilation process we use the GNU Make utility, a standard build automation tool that automatically builds the executable program. 
You will want to check the 
config folder. Here, machine specific Makefile variables are defined. 
The default.mk file gives an example. You can use the default 
if you create an include folder in your home directory and create 
links to the aforementioned libraries. Or you can 
create your own file. 
#### 4. Documentation 
The documentation can be generated with [Doxygen](http://www.doxygen.org) in each subfolder or you can access the whole documentation [online](http://feltor-dev.github.io/feltor/inc/dg/html/index.html).
#### 5. Official releases 
Our latest code release has a shiny DOI badge from zenodo

[![DOI](https://zenodo.org/badge/14143578.svg)](https://zenodo.org/badge/latestdoi/14143578)

which makes us officially citable.
