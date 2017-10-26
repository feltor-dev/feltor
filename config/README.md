The Makefiles in FELTOR are configured by hardware specific *.mk files in the config folder. Every Makefile includes these files in the order  

```shell
feltor/config/default.mk #Sets the default variables
feltor/config/*.mk       #overwrite variables if machine is recognized
feltor/config/devices/devices.mk    #recombine variables depending on device
```

The machine specific config files (e.g. vsc3.mk) should have an include guard and can overwrite or add to any of the following variables:

| variable  | default value                            | description                              |
| :-------: | :--------------------------------------- | :--------------------------------------- |
|    CC     | g++                                      | C++ compiler                             |
|  CFLAGS   | -std=c++11 -Wall -x c++                  | flags for the C++ compiler               |
|   MPICC   | mpic++                                   | the mpi wrapper for the c++ compiler     |
| MPICFLAGS | -std=c++11 -Wall -x c++                  | flags for MPI compilation                |
|   NVCC    | nvcc                                     | CUDA compiler                            |
| NVCCFLAGS | -std=c++11                               | flags for nvcc                           |
| NVCCARCH  | -arch sm_20                              | specify the **gpu** compute capability  https://developer.nvidia.com/cuda-gpus (can be overwritten on the command line) |
|    OPT    | -O3                                      | optimization flags for the **host** code (can be overwritten on the command line, CUDA kernel code is always compiled with -O3) |
|  OMPFLAG  | -fopenmp                                 | The compiler flag activating the OpenMP support |
|           |                                          |                                          |
|  INCLUDE  | -I$(HOME)/include                        | cusp, thrust, json and the draw libraries. The default expects to find (symbolic links to ) these libraries in your home folder |
|   LIBS    | -lnetcdf -lhdf5 -ldhf5_hl                | netcdf library                           |
|  JSONLIB  | -L$(HOME)/include/json/../../src/lib_json -ljsoncpp | the JSONCPP library                      |
|  GLFLAGS  | $$(pkg-config --static --libs glfw3)     | glfw3 installation (if glfw3 was installed correctly the default should work) |


The file `feltor/config/devices/devices.mk` defines device specific configurations and MACROS that essentially steer the behaviour of the cusp and thrust libraries and serve as include guards. These are activated by setting the variable **device**, which for now can take one of the following values:

| value | description                              | flags                                    |
| ----- | ---------------------------------------- | ---------------------------------------- |
| gpu   | combines CC and NVCC  into CC, CFLAGS, NVCCFLAGS and NVCCARCH  into CFLAGS and analogously the MPI flags MPICC and NVCC into MPICC, MPICFLAGS, NVCCFLAGS and NVCCARCH into MPICFLAGS using nvcc's --compiler-bindir and -Xcompiler options | -D_FORCE_INLINES added to CFLAGS and MPICFLAGS |
| !gpu  | if device != gpu all thrust device calls redirect to OpenMP using THRUST_DEVICE_SYSTEM macro | -DTHRUST_DEVICE_SYSTEM= THRUST_DEVICE_SYSTEM_OMP $(OMPFLAG) added to both CFLAGS and MPICFLAGS |
| mic   | specify OPT for Intel Xeon Phi architecture | OPT = -O3 -xMIC-AVX512                   |
| skl   | specify OPT for Intel Skylake processors | OPT = -xCORE-AVX512 -mtune=skylake -O3   |

### Examples

The **device** variable should be, the **OPT** and the **NVCCARCH** variables are intended to be specified on the command line: 

```shell
make blas_b device=gpu NVCCARCH=-arch sm_35 #Compile using nvcc for a Tesla K40
make blas_b device=omp OPT=-O2              #Compile for OpenMP using -O2 
```

```shell
make blas_mpib device=mic #Compile an MPI program for the Xeon Phi architecture 
```

