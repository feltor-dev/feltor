MPICXX=mpic++
CXXFLAGS=#-Wall
CXXFLAGS+= -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP

%_mpit: %_mpit.cpp 
	$(MPICXX) $(INCLUDE) -DDG_DEBUG $(CXXFLAGS) $< -o $@

%_mpib: %_mpib.cpp
	$(MPICXX) -O2 $(CXXFLAGS) $< -o $@ $(INCLUDE) 
