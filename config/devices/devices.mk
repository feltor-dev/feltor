ifeq ($(strip $(device)),gpu)
ccc_:=$(CC)
CC = $(NVCC) --compiler-bindir $(ccc_)
CFLAGS = $(NVCCARCH) $(NVCCFLAGS) 
CFLAGS+=-D_FORCE_INLINES # solves issue with std=c++11
CFLAGS+=-D_MWAITXINTRIN_H_INCLUDED # solves issue with std=c++11
############################################
mpiccc_:=$(MPICC)
MPICC=nvcc --compiler-bindir $(mpiccc_)
MPICFLAGS+= $(NVCCARCH) $(NVCCFLAGS)
MPICFLAGS+=-D_FORCE_INLINES # solves issue with std=c++11
MPICFLAGS+=-D_MWAITXINTRIN_H_INCLUDED # solves issue with std=c++11
else
CFLAGS+=-x c++ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP $(OMPFLAG)
MPICFLAGS+=$(CFLAGS)
endif #device=gpu
ifeq ($(strip $(device)),omp)
endif #device=omp
ifeq ($(strip $(device)),knl)
OPT=-O3 -xMIC-AVX512 
#OPT=-O3 -mavx512er
endif #device=mic
ifeq ($(strip $(device)),skl)
OPT=-xCORE-AVX512 -mtune=skylake -O3 
#OPT=-O3 -mtune=skylake
endif #device=mic
