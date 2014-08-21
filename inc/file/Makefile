CXX=g++
MPICXX=mpic++
system = home

INCLUDE =-I../
INCLUDE += -I$(HOME)/include

CFLAGS = #-Wall -std=c++0x 
CFLAGS+= -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
LIBS = -lnetcdf

ifeq ($(strip $(system)),leo3)
INCLUDE += -I$(HOME)/include
INCLUDE += -I$(UIBK_HDF5_INC)
INCLUDE += -I$(UIBK_OPENMPI_INC)

LIBS 	 = -L$(UIBK_HDF5_LIB) -lhdf5 -lhdf5_hl 
LIBS 	+= -L$(HOME)/lib -lnetcdf -lcurl -lm
GLFLAGS  = -lm 
endif

all: read_input_t netcdf_t netcdf_mpit

read_input_t: read_input_t.cpp read_input.h 
	$(CXX) $< -o $@ 


netcdf_t: netcdf_t.cpp nc_utilities.h
	$(CXX) $< -o $@ $(CFLAGS) -g $(INCLUDE) $(LIBS) 

netcdf_mpit: netcdf_mpit.cpp nc_utilities.h
	$(MPICXX) $< -o $@ $(CFLAGS) $(INCLUDE) $(LIBS) 

.PHONY: doc clean

doc:
	doxygen Doxyfile


clean:
	rm -f read_input_t netcdf_t netcdf_mpit
