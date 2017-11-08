ifeq ($(strip $(device)),gpu)
ccc_:=$(CC)
CC = nvcc --compiler-bindir $(ccc_)
OPT=-O3
CFLAGS+=-D_FORCE_INLINES
CFLAGS+= --compiler-options -Wall $(NVCCARCH)
#CFLAGS+= -Xcompiler $(OMPFLAG)
#CFLAGS+= -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP
#CFLAGS+= -DCUSP_DEVICE_BLAS_SYSTEM=CUSP_DEVICE_BLAS_CUBLAS -lcublas
#CFLAGS+= -DCUSP_USE_TEXTURE_MEMORY
mpiccc_:=$(MPICC)
MPICC=nvcc --compiler-bindir $(mpiccc_)
MPICFLAGS+= -D_FORCE_INLINES
#MPICFLAGS+= -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
#MPICFLAGS+= -DCUSP_DEVICE_BLAS_SYSTEM=CUSP_DEVICE_BLAS_CUBLAS -lcublas
#MPICFLAGS+= -DCUSP_USE_TEXTURE_MEMORY
MPICFLAGS+= --compiler-options -Wall $(NVCCARCH)
MPICFLAGS+= --compiler-options $(OPT)
endif #device=gpu
ifeq ($(strip $(device)),omp)
CFLAGS+=-Wall -x c++
CFLAGS+= -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
CFLAGS+= $(OMPFLAG) 
MPICFLAGS+=$(CFLAGS) #includes values in CFLAGS defined later
endif #device=omp
ifeq ($(strip $(device)),mic)
CFLAGS+=-Wall -x c++ 
CFLAGS+= -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
CFLAGS+= $(OMPFLAG) #-mmic 
MPICFLAGS+=$(CFLAGS) #includes values in CFLAGS defined later
OPT=-O3 -xMIC-AVX512 
endif #device=mic
ifeq ($(strip $(device)),skl)
CFLAGS+=-Wall -x c++ 
CFLAGS+= -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
CFLAGS+= $(OMPFLAG) #-mmic 
MPICFLAGS+=$(CFLAGS) #includes values in CFLAGS defined later
OPT=-xCORE-AVX512 -mtune=skylake -O3 
endif #device=mic
