ifeq ($(strip $(device)),gpu)
ccc_:=$(CC)
CC = $(NVCC) --compiler-bindir $(ccc_)
flags_:=$(CFLAGS)
CFLAGS = -Xcompiler "$(flags_)"
CFLAGS+= $(NVCCARCH) $(NVCCFLAGS) 
CFLAGS+=-D_FORCE_INLINES
############################################
mpiccc_:=$(MPICC)
MPICC=nvcc --compiler-bindir $(mpiccc_)
mpiflags_:=$(MPICFLAGS)
MPICFLAGS = -Xcompiler "$(mpiflags_)" $(NVCCARCH) $(NVCCFLAGS)
MPICFLAGS+= -D_FORCE_INLINES
else
CFLAGS+= -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP $(OMPFLAG)
MPICFLAGS+= -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP $(OMPFLAG)
endif #device=gpu
ifeq ($(strip $(device)),omp)
endif #device=omp
ifeq ($(strip $(device)),mic)
OPT=-O3 -xMIC-AVX512 
endif #device=mic
ifeq ($(strip $(device)),skl)
OPT=-xCORE-AVX512 -mtune=skylake -O3 
endif #device=mic
