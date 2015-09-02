
ifeq ($(strip $(HPC_SYSTEM)),vsc2)
INCLUDE += -I$(HOME)/include
INCLUDE += -I/opt/intel/impi/4.1.0/include64
INCLUDE += -I/opt/sw/netcdf/4.3.2/include
INCLUDE += -I/opt/hdf5/1.8.9/intel/include
GLFLAGS  = -lm
CC=icc
MPICC=mpiicc
OPT=-O3
MPICFLAGS  += -DMPICH_IGNORE_CXX_SEEK
LIBS     = -L/opt/hdf5/1.8.9/intel/lib -lhdf5 -lhdf5_hl
LIBS    += -L/opt/intel/composerxe/lib/intel64 -lirc -lsvml
LIBS    += -L/opt/intel/impi/4.1.0/lib64 -lmpi
LIBS    += -L/opt/sw/netcdf/4.3.2/lib -lnetcdf -lcurl
endif
