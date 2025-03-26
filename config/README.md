### Configuration

The Makefiles for programs in FELTOR are configured by hardware specific *.mk files in the config folder. Every Makefile includes these files in the order

```shell
feltor/config/default.mk            #Defines the default variables
feltor/config/*.mk                  #overwrite variables if machine is recognized
feltor/config/devices/devices.mk    #recombine variables depending on device
```

You can create your own machine specific config file e.g. `feltor/config/your-machine.mk` **Please do not commit unique machine specific config files**.
Through the above construct the config file will be included after `default.mk` and before `devices.mk`.
To avoid inclusion on machines other than your own the file should have an include guard; something like
```shell
ifeq ($(strip $(shell domainname)),leo3-domain)
# ...
endif
```

Within the file you can overwrite or add to any of the following variables:

| variable  | default value                | description                                                  |
| :-------: | :--------------------------- | :----------------------------------------------------------- |
|    CC     | g++                          | C++ compiler                                                 |
|   MPICC   | mpic++                       | the corresponding mpi wrapper for the c++ compiler           |
|  CFLAGS   | -std=c++17 -mavx -mfma -Wall | flags for the C++ compiler, avx and fma are recommended if the CPU supports it |
| MPICFLAGS |                              | flags specific to the MPI compilation                        |
|   OPT    | -O2                                      | optimization flags for the **host** code (can be overwritten on the command line, CUDA kernel code is always compiled with -O3) |
|  OMPFLAG  | -fopenmp                                 | The compiler flag activating the OpenMP support |
|   NVCC    | nvcc                                     | CUDA compiler                            |
| NVCCFLAGS | -std=c++17  -Xcompiler "-Wall -mavx -mfma"                             | flags for nvcc  and underlying host compiler, (minimum instruction set is sse4.1, avx and fma are recommended)                         |
| NVCCARCH  | -arch sm_61                              | specify the **gpu** compute capability  https://developer.nvidia.com/cuda-gpus (note: can be overwritten on the command line) |
|  INCLUDE  | -I$(HOME)/include                        | thrust, cub, cuda, nv vcl and the draw (if needed) libraries. The default expects to find (symbolic links to ) these libraries in your home/include folder |
|   LIBS    | -lnetcdf -lhdf5_serial -ldhf5_serial_hl                | netcdf and hdf5 library. Normally, we would use -lhdf5 and -lhdf5_hl but the libhdf5-dev package installs them with "_serial". |
|  JSONLIB  | -ljsoncpp          | the JSON library, the default is assumed to be jsoncpp; if instead `-DDG_USE_JSONHPP` is set then the (header-only) nlohmann-json library is used                       |
|  LAPACKLIB  | -llapacke | the lapack library                      |
|  GLFLAGS  | $$(pkg-config --static --libs glfw3)     | glfw3 installation (if glfw3 was installed correctly the default should work) |


The main purpose of the file `feltor/config/devices/devices.mk` is to configure the nvcc + X compilation but it can also be used to specifiy optimizations for specific hardware. These are activated by setting the variable **device**, which for now can take one of the following values:

| value | description                                                  | flags                                                        |
| ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| gpu   | replaces the CC and CFLAGS variables with the nvcc versions and analogously MPICC and MPICFLAGS, Add `-lcusparse` to `LIBS` | `CC = $(NVCC) -x cu --compiler-bindir $(CC)` `CFLAGS = $(NVCCARCH) $(NVCCFLAGS)` `MPICC = $(NVCC) -x cu --compiler-bindir $(MPICC)` `MPICFLAGS+= $(NVCCARCH) $(NVCCFLAGS)` |
| omp   | all thrust device calls redirect to OpenMP using the THRUST_DEVICE_SYSTEM macro | `CFLAGS += -x c++ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP  $(OMPFLAG)`, `MPICFLAGS+=$(CFLAGS)` |
| cpu   | all thrust device calls redirect to single thread version using the THRUST_DEVICE_SYSTEM macro | `CFLAGS += -x c++ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP`, `MPICFLAGS+=$(CFLAGS)` |
| knl   | same as omp but specifies OPT for Intel Xeon Phi architecture | `OPT = -O3 -xMIC-AVX512`                                     |
| skl   | same as omp but specifies OPT for Intel Skylake processors (icc only) | `OPT = -xCORE-AVX512 -mtune=skylake -O3`                     |
#### Cuda aware MPI
If one compiles for the MPI+GPU backend, the dg library **by default assumes the MPI library is cuda-aware**. If this is not the case one has to set the Macro `-DDG_CUDA_UNAWARE_MPI` during compilation. With OpenMPI this is unnecessary as we can automatically test if the MPI library is cuda-aware but unfortunately not for any other MPI library. In any case the dg library defines the constexpr boolean value `dg::cuda_aware_mpi`. 
### Examples

The **device** variable should be, the **OPT** and the **NVCCARCH** variables can be specified on the command line:

```shell
#Compile using nvcc for a Tesla K40:
make blas_b device=gpu NVCCARCH='-arch sm_61'

#Compile for OpenMP using -O2:
make blas_b device=omp OPT=-O2

#Hybrid MPI+OpenMP program for the Xeon Phi architecture:
make blas_mpib device=knl

#Hybrid MPI+GPU program for the Tesla P100 GPU, host code with -O2:
make blas_mpib device=gpu NVCCARCH='-arch sm_60' OPT=-O2
```

### General Remarks
 - If MPI is used in connection with the gpu backend, the mpi installation would ideally be **cuda-aware** but does not need to be
 - If `icc` is used as the C++ compiler the `-restrict` option has to be used to enable the recognition of the restrict keyword
 - Support for OpenMP-4 is recommended (at least gcc-4.9 or icc-15), but not mandatory
 - The library headers are compliant with the c++17 standard but we reserve the right to upgrade that in future updates
 - For a mac, we need X-code to compile. At the same time, we need to include the paths to the json, hdf5 and netcdf libraries. We work without GLFW, as it gives problems.  As it is now, the libraries are installed with homebrew, one of the prefered installers for mac. If other programs are used, it is neccessary to include the paths to the libraries used. We compile with g++, but other compilers might also work (like clang++).

