
ifeq ($(strip $(HPC_SYSTEM)),vsc3)
INCLUDE += -I$(HOME)/include
INCLUDE += -I/cm/shared/apps/intel/impi_5.0.3/intel64/include
INCLUDE += -I/opt/sw/x86_64/glibc-2.12/ivybridge-ep/netcdf/4.3.2/intel-14.0.2/include
INCLUDE += -I/opt/sw/x86_64/glibc-2.12/ivybridge-ep/hdf5/1.8.12/intel-14.0.2/include
GLFLAGS  = -lm
CC=icc
MPICC=mpiicc
OPT=-O3
MPICFLAGS+= -DMPICH_IGNORE_CXX_SEEK
OMPFLAG=-openmp
LIBS    +=-L/opt/sw/x86_64/glibc-2.12/ivybridge-ep/hdf5/1.8.12/intel-14.0.2/lib -lhdf5 -lhdf5_hl
LIBS    +=-L/cm/shared/apps/intel-cluster-studio/composer_xe_2013_sp1.2.144/compiler/lib/intel64 -lirc -lsvml
LIBS    += -L/cm/shared/apps/intel/impi_5.0.3/intel64/lib -lmpi
LIBS    +=-L/opt/sw/x86_64/glibc-2.12/ivybridge-ep/netcdf/4.3.2/intel-14.0.2/lib -lnetcdf -lcurl
endif
