ifeq ($(strip $(device)),gpu)
ccc_:=$(CC)
CC = $(NVCC) --compiler-bindir $(ccc_)
CFLAGS = -x cu $(NVCCARCH) $(NVCCFLAGS)
# 21.1.25 commented out next 2 lines
# Seems to work, shouldn't have any performance issues!?
#CFLAGS+=-D_FORCE_INLINES # solves issue with std=c++11
#CFLAGS+=-D_MWAITXINTRIN_H_INCLUDED # solves issue with std=c++11
############################################
mpiccc_:=$(MPICC)
MPICC=$(NVCC) --compiler-bindir $(mpiccc_)
MPICFLAGS+= -x cu $(NVCCARCH) $(NVCCFLAGS)
#MPICFLAGS+=-D_FORCE_INLINES # solves issue with std=c++11
#MPICFLAGS+=-D_MWAITXINTRIN_H_INCLUDED # solves issue with std=c++11
endif #device=gpu

ifeq ($(strip $(device)),cpu)
CFLAGS+=-x c++ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
MPICFLAGS+=$(CFLAGS)
endif #device=cpu

ifeq ($(strip $(device)),omp)
CC+=$(OMPFLAG)
MPICC+=$(OMPFLAG)
CFLAGS+=-x c++ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
MPICFLAGS+=$(CFLAGS)
endif #device=omp

ifeq ($(strip $(device)),knl)
CC+=$(OMPFLAG)
MPICC+=$(OMPFLAG)
CFLAGS+=-x c++ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
MPICFLAGS+=$(CFLAGS)
OPT=-O3 -xMIC-AVX512
endif #device=mic

ifeq ($(strip $(device)),skl)
CC+=$(OMPFLAG)
MPICC+=$(OMPFLAG)
CFLAGS+=-x c++ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
MPICFLAGS+=$(CFLAGS)
OPT=-xCORE-AVX512 -O3 # -mtune=skylake # do not use mtune according to cineca
endif #device=mic
