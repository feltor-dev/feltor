
ifeq ($(strip $(shell domainname)),leo3-domain)
INCLUDE  = -I$(HOME)/include
INCLUDE += -I$(UIBK_HDF5_INC)
INCLUDE += -I$(UIBK_OPENMPI_INC)
INCLUDE += -I$(UIBK_NETCDF_4_INC)
GLFLAGS  = -lm
CC=g++
MPICC=mpicxx
OPT=-O3
NVCCARCH=-arch=sm_35 # fermi cards
OMPFLAG=-fopenmp
LIBS 	 = -L$(UIBK_HDF5_LIB) -lhdf5 -lhdf5_hl 
LIBS 	+= -L$(UIBK_NETCDF_4_LIB) -lnetcdf -lcurl -lm
endif
