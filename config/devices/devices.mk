ifeq ($(strip $(device)),gpu)
ccc_:=$(CC)
CC = $(NVCC) -x cu --compiler-bindir $(ccc_)
CFLAGS = $(NVCCARCH) $(NVCCFLAGS)
CFLAGS+=-D_FORCE_INLINES # solves issue with std=c++11
CFLAGS+=-D_MWAITXINTRIN_H_INCLUDED # solves issue with std=c++11
############################################
mpiccc_:=$(MPICC)
MPICC=$(NVCC) -x cu --compiler-bindir $(mpiccc_)
MPICFLAGS+= $(NVCCARCH) $(NVCCFLAGS)
MPICFLAGS+=-D_FORCE_INLINES # solves issue with std=c++11
MPICFLAGS+=-D_MWAITXINTRIN_H_INCLUDED # solves issue with std=c++11
endif #device=gpu

ifeq ($(strip $(device)),cpp)
CFLAGS+=-x c++ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
MPICFLAGS+=$(CFLAGS)
endif #device=cpp
ifeq ($(strip $(device)),cpu)
CFLAGS+=-x c++ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
MPICFLAGS+=$(CFLAGS)
endif #device=cpu

ifeq ($(strip $(device)),omp)
CFLAGS+=-x c++ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP $(OMPFLAG)
MPICFLAGS+=$(CFLAGS)
endif #device=omp

ifeq ($(strip $(device)),knl)
CFLAGS+=-x c++ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP $(OMPFLAG)
MPICFLAGS+=$(CFLAGS)
OPT=-O3 -xMIC-AVX512
endif #device=mic

ifeq ($(strip $(device)),skl)
CFLAGS+=-x c++ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP $(OMPFLAG)
MPICFLAGS+=$(CFLAGS)
OPT=-xCORE-AVX512 -O3 # -mtune=skylake # do not use mtune according to cineca
endif #device=mic


ifeq ($(strip $(device)),mac_cpu)
CC=g++
CFLAGS=-Wall -std=c++14
CFLAGS+=-x c++ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
LIBS=-L/opt/homebrew/Cellar/netcdf/4.8.1_3/lib -lnetcdf -L/opt/homebrew/Cellar/hdf5/1.12.2_1/lib -lhdf5 -lhdf5_hl
INCLUDE+= -I/opt/homebrew/Cellar/netcdf/4.8.1_3/include # To add LLVM and see if it solves compilator problems
JSONLIB= -L/opt/homebrew/Cellar/jsoncpp/1.9.5/lib -ljsoncpp
INCLUDE+= -I/opt/homebrew/Cellar/jsoncpp/1.9.5/include
GLFLAGS =-DWITHOUT_GLFW#$$(pkg-config --static --libs glfw3 gl) #-lglfw
endif #device=cpu
