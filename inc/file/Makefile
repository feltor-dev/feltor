CXX=g++
system = home

INCLUDE =-I../
INCLUDE+=-I$(HOME)/include

CFLAGS = -Wall -std=c++0x 
LIBS = -lnetcdf

ifeq ($(strip $(system)),leo3)
INCLUDE += -I$(HOME)/include
INCLUDE += -I$(UIBK_NETCDF_INC)
#INCLUDE += -I$(UIBK_OPENMPI_INC)

LIBS 	 = -L$(UIBK_NETCDF_LIB) -lnetcdf
GLFLAGS  = -lm
endif


read_input_t: read_input_t.cpp read_input.h 
	$(CXX) $< -o $@ 


netcdf_t: netcdf_t.cpp nc_utilities.h
	$(CXX) $< -o $@ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP  $(INCLUDE) $(LIBS) 


.PHONY: doc clean

doc:
	doxygen Doxyfile


clean:
	rm -f read_input_t
