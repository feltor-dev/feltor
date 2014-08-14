MPICXX=mpic++
CXXFLAGS=#-Wall
CXXFLAGS+= -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
LIBS = -lnetcdf

%_mpit: %_mpit.cpp 
	$(MPICXX) $(INCLUDE) -DDG_DEBUG $(CXXFLAGS) $< -o $@

%_mpib: %_mpib.cpp
	$(MPICXX) -O3 $(CXXFLAGS) $< -o $@ $(INCLUDE) 
	

toefl_mpib: toefl_mpib.cpp toefl.cuh
	$(MPICXX) -O3 $(CXXFLAGS) $< -o $@ $(INCLUDE) $(LIBS)
