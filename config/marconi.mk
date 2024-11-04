
ifeq ($(strip $(HPC_SYSTEM)),marconi)
CC=icc
MPICC=mpiicc -mt_mpi
OPT=-O3 -xHost  # overwritten for skl in devices.mk
#MPICFLAGS+= -DMPICH_IGNORE_CXX_SEEK
OMPFLAG=-qopenmp
CFLAGS=-Wall -std=c++17 -restrict -fp-model precise -fimf-arch-consistency=true #-mfma  #flags for CC

INCLUDE += -I$(HOME)/include # cusp, thrust
INCLUDE += -I$(NETCDF_INC) -I$(HDF5_INC) -I$(JSONCPP_INC)
JSONLIB=-L$(JSONCPP_LIB) -ljsoncpp
LIBS    =-L$(HDF5_LIB) -lhdf5 -lhdf5_hl
LIBS    +=-L$(NETCDF_LIB) -lnetcdf -lcurl
endif
#############################modules to load in .bashrc#######################
#module load profile/advanced
#module load intel/pe-xe-2018--binary
#module load intelmpi/2018--binary
#module load szip/2.1--gnu--6.1.0
#module load zlib/1.2.8--gnu--6.1.0
#module load hdf5/1.10.4--intel--pe-xe-2018--binary
#module load netcdf/4.6.1--intel--pe-xe-2018--binary
#module load gnu/8.3.0
#module load jsoncpp/1.9.3--gnu--8.3.0
