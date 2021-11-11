ifeq ($(strip $(HPC_SYSTEM)),m100)
#CC=xlc++ #C++ compiler
#MPICC=mpixlC  #mpi compiler
#CFLAGS=-Wall -std=c++1y -DWITHOUT_VCL -mcpu=power9 -qstrict# -mavx -mfma #flags for CC
#OMPFLAG=-qsmp=omp
CFLAGS=-Wall -std=c++14 -DWITHOUT_VCL -mcpu=power9 # -mavx -mfma #flags for CC
OPT=-O3 # optimization flags for host code
NVCC=nvcc #CUDA compiler
NVCCARCH=-arch sm_70 -Xcudafe "--diag_suppress=code_is_unreachable --diag_suppress=initialization_not_reachable" #nvcc gpu compute capability
NVCCFLAGS= -std=c++14 -Xcompiler "-mcpu=power9 -Wall" --extended-lambda# -mavx -mfma" #flags for NVCC

INCLUDE += -I$(NETCDF_INC) -I$(HDF5_INC) -I$(JSONCPP_INC)
INCLUDE += -L$(BOOST_INC)
JSONLIB=-L$(JSONCPP_LIB) -ljsoncpp # json library for input parameters
LIBS    =-L$(HDF5_LIB) -lhdf5 -lhdf5_hl
LIBS    +=-L$(NETCDF_LIB) -lnetcdf -lcurl
LIBS    +=-L$(BOOST_LIB)
endif
#########################Modules to load ##################
#module load profile/base
#module load cuda/11.0
#module load gnu/8.4.0
#module load spectrum_mpi/10.3.1--binary
#module load binutils/2.34
#module load zlib/1.2.11--gnu--8.4.0
#module load szip/2.1.1--gnu--8.4.0
#module load hdf5/1.12.0--gnu--8.4.0
#module load netcdf/4.7.3--gnu--8.4.0
#module load jsoncpp/1.9.3--spectrum_mpi--10.3.1--binary
#module load boost/1.72.0--spectrum_mpi--10.3.1--binary
#module load blas/3.8.0--gnu--8.4.0
#module load lapack/3.9.0--gnu--8.4.0


