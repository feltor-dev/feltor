# if device = cpu
MPICXX=mpic++ -x c++
CXXFLAGS =-Wall
CXXFLAGS+= -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
LIBS = -lnetcdf
ifeq ($(strip $(device)),omp)
CXXFLAGS = -fopenmp -Wall -x c++
CXXFLAGS+= -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
endif
ifeq ($(strip $(device)),gpu)
MPICXX=nvcc --compiler-bindir mpic++
CXXFLAGS = -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
CXXFLAGS+= --compiler-options -Wall -arch=sm_20 
endif

%_mpit: %_mpit.cu 
	$(MPICXX) $(INCLUDE) -DDG_DEBUG $(CXXFLAGS) $< -o $@ -g

%_mpib: %_mpib.cu
	$(MPICXX) -O3 $(CXXFLAGS) $< -o $@ $(INCLUDE) 
	

toefl_mpib: toefl_mpib.cu toefl.cuh
	$(MPICXX) -O3 $(CXXFLAGS) $< -o $@ $(INCLUDE) $(LIBS) -DDG_BENCHMARK
