CXX=g++

INCLUDE =-I/home/matthias/include
INCLUDE+=-I../


read_input_t: read_input_t.cpp read_input.h 
	$(CXX) $< -o $@ 


netcdf_t: netcdf_t.cpp nc_utilities.h
	$(CXX) $< -o $@ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP  $(INCLUDE) -lnetcdf


.PHONY: doc clean

doc:
	doxygen Doxyfile


clean:
	rm -f read_input_t
