device=omp
#configure machine
include ../../config/default.mk
include ../../config/*.mk
include ../../config/devices/devices.mk

INCLUDE+= -I../../ # other project libraries
INCLUDE+= -I../    # other project libraries

all: netcdf_t netcdf_mpit json_utilities_t

netcdf_t: netcdf_t.cpp nc_utilities.h easy_output.h
	$(CC) $< -o $@ $(CFLAGS) -g $(INCLUDE) $(LIBS)

netcdf_mpit: netcdf_mpit.cpp nc_utilities.h easy_output.h
	$(MPICC) $< -o $@ $(MPICFLAGS) $(INCLUDE) $(LIBS)

json_utilities_t: json_utilities_t.cpp json_utilities.h
	$(CC) $< -o $@ $(CFLAGS) -g $(INCLUDE) $(JSONLIB)

.PHONY: doc clean

doc:
	doxygen Doxyfile

clean:
	rm -f netcdf_t netcdf_mpit
