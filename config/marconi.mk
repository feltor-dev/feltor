
ifeq ($(strip $(HPC_SYSTEM)),marconi)
INCLUDE += -I$(HOME)/include # cusp, thrust
INCLUDE += -I$(NETCDF_INC) -I$(HDF5_INC)
GLFLAGS  = -lm
CC=icc
MPICC=mpiicc
OPT=-O3 -xHost
#MPICFLAGS+= -DMPICH_IGNORE_CXX_SEEK
OMPFLAG=-qopenmp
JSONLIB=-L$(HOME)/include/json/../../src/lib_json -ljsoncpp # json library for input parameters
LIBS    +=-L$(HDF5_LIB) -lhdf5 -lhdf5_hl
LIBS    +=-L$(NETCDF_LIB) -lnetcdf -lcurl
endif
#############################modules to load in .bashrc#######################
#module load profile/base                         
#module load intel/pe-xe-2017--binary             
#module load intelmpi/2017--binary                
#module load szip/2.1--gnu--6.1.0                 
#module load zlib/1.2.8--gnu--6.1.0               
#module load hdf5/1.8.17--intelmpi--2017--binary  
#module load netcdf/4.4.1--intelmpi--2017--binary 




