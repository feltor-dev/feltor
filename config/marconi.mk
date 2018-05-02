
ifeq ($(strip $(HPC_SYSTEM)),marconi)
CC=icc
MPICC=mpiicc -mt_mpi
OPT=-O3 -xHost  # overwritten for mic in devices.mk
#MPICFLAGS+= -DMPICH_IGNORE_CXX_SEEK
OMPFLAG=-qopenmp
CFLAGS=-Wall -std=c++11 -restrict -fp-model precise -fimf-arch-consistency=true #-mfma  #flags for CC

INCLUDE += -I$(HOME)/include # cusp, thrust
INCLUDE += -I$(NETCDF_INC) -I$(HDF5_INC)
LIBS    +=-L$(HDF5_LIB) -lhdf5 -lhdf5_hl
LIBS    +=-L$(NETCDF_LIB) -lnetcdf -lcurl
endif
#Using GNU compiler
#ifeq ($(strip $(HPC_SYSTEM)),marconi)
#INCLUDE += -I$(HOME)/include # cusp, thrust
#INCLUDE += -I$(NETCDF_INC) -I$(HDF5_INC)
#LIBS    +=-L$(HDF5_LIB) -lhdf5 -lhdf5_hl
#LIBS    +=-L$(NETCDF_LIB) -lnetcdf -lcurl
#endif
#############################modules to load in .bashrc#######################
#module load profile/base                         
#module load intel/pe-xe-2017--binary             
#module load intelmpi/2017--binary                
#module load szip/2.1--gnu--6.1.0                 
#module load zlib/1.2.8--gnu--6.1.0               
#module load hdf5/1.8.17--intelmpi--2017--binary  
#module load netcdf/4.4.1--intelmpi--2017--binary 


###########configure mic jobs#########################
#export KMP_AFFINITY=scatter #important
#srun --partition=knl_fua_prod --constraint="cache" --qos=knl_qos_fuadbg --account=FUA22_FELTOR --nodes=1 --time=0:30:00 --pty /bin/bash
