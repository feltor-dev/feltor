### Configuration

The Makefiles for programs in FELTOR are configured by hardware specific *.mk files in the config folder. Every Makefile includes these files in the order

```shell
feltor/config/default.mk            #Defines the default variables
feltor/config/*.mk                  #overwrite variables if machine is recognized
feltor/config/devices/devices.mk    #recombine variables depending on device
```

Your machine specific config file (e.g. feltor/config/your-machine.mk) should have an include guard and overwrite or add to any of the following variables:

| variable  | default value                | description                                                  |
| :-------: | :--------------------------- | :----------------------------------------------------------- |
|    CC     | g++                          | C++ compiler                                                 |
|   MPICC   | mpic++                       | the corresponding mpi wrapper for the c++ compiler           |
|  CFLAGS   | -std=c++14 -mavx -mfma -Wall | flags for the C++ compiler, avx and fma are recommended if the CPU supports it |
| MPICFLAGS |                              | flags specific to the MPI compilation                        |
   OPT    | -O3                                      | optimization flags for the **host** code (can be overwritten on the command line, CUDA kernel code is always compiled with -O3) |
|  OMPFLAG  | -fopenmp                                 | The compiler flag activating the OpenMP support |
|   NVCC    | nvcc                                     | CUDA compiler                            |
| NVCCFLAGS | -std=c++14  -Xcompiler "-Wall -mavx -mfma"                             | flags for nvcc  and underlying host compiler, (minimum instruction set is sse4.1, avx and fma are recommended)                         |
| NVCCARCH  | -arch sm_61                              | specify the **gpu** compute capability  https://developer.nvidia.com/cuda-gpus (note: can be overwritten on the command line) |
|                                          |                                          |     |
|  INCLUDE  | -I$(HOME)/include                        | cusp, thrust, json, vcl and the draw (if needed) libraries. The default expects to find (symbolic links to ) these libraries in your home folder |
|   LIBS    | -lnetcdf -lhdf5 -ldhf5_hl                | netcdf library                           |
|  JSONLIB  | -L$(HOME)/include/json/../../src/lib_json -ljsoncpp | the JSONCPP library                      |
|  LAPACKLIB  | -llapacke | the lapack library                      |
|  GLFLAGS  | $$(pkg-config --static --libs glfw3)     | glfw3 installation (if glfw3 was installed correctly the default should work) |


The main purpose of the file `feltor/config/devices/devices.mk` is to configure the nvcc + X compilation but it can also be used to specifiy optimizations for specific hardware. These are activated by setting the variable **device**, which for now can take one of the following values:

| value | description                              | flags                                    |
| ----- | ---------------------------------------- | ---------------------------------------- |
| gpu   | replaces the CC and CFLAGS variables with the nvcc versions and analogously MPICC and MPICFLAGS | `CC = $(NVCC) --compiler-bindir $(CC)` `CFLAGS = $(NVCCARCH) $(NVCCFLAGS)` `MPICC = $(NVCC) --compiler-bindir $(MPICC)` `MPICFLAGS+= $(NVCCARCH) $(NVCCFLAGS)` |
| !gpu  | if device != gpu all thrust device calls redirect to OpenMP using the THRUST_DEVICE_SYSTEM macro | `-x c++` `-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP` and `$(OMPFLAG)` added to CFLAGS, `MPICFLAGS+=$(CFLAGS)` |
| omp   | specify OPT for OpenMP                   | OPT = -O3                                |
| mic   | specify OPT for Intel Xeon Phi architecture | OPT = -O3 -xMIC-AVX512                   |
| skl   | specify OPT for Intel Skylake processors (icc only) | OPT = -xCORE-AVX512 -mtune=skylake -O3   |

### Examples

The **device** variable should be, the **OPT** and the **NVCCARCH** variables can be specified on the command line:

```shell
#Compile using nvcc for a Tesla K40:
make blas_b device=gpu NVCCARCH='-arch sm_61'

#Compile for OpenMP using -O2:
make blas_b device=omp OPT=-O2

#Hybrid MPI+OpenMP program for the Xeon Phi architecture:
make blas_mpib device=mic

#Hybrid MPI+GPU program for the Tesla P100 GPU, host code with -O2:
make blas_mpib device=gpu NVCCARCH='-arch sm_60' OPT=-O2
```

### General Remarks
 - If MPI is used in connection with the gpu backend, the mpi installation would ideally be **cuda-aware** but does not need to be
 - If `icc` is used as the C++ compiler the `-restrict` option has to be used to enable the recognition of the restrict keyword
 - Support for OpenMP-4 is recommended (at least gcc-4.9 or icc-15), but not mandatory
 - The library headers are compliant with the c++14 standard but we reserve the right to upgrade that in future updates


