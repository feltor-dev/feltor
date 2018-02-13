
ifeq ($(strip $(shell domainname)),psg.cluster.zone)
INCLUDE = -I$(HOME)/include
GLFLAGS  = -lm
#CC=pgc++
CC=g++
MPICC=mpic++
OPT=-O3
#MPICFLAGS+= -DMPICH_IGNORE_CXX_SEEK
OMPFLAG=-fopenmp# flag for CC and MPICC
#JSONLIB=-L$(HOME)/include/json/../../src/lib_json -ljsoncpp # json library for input parameters
NVCCARCH=-arch sm_60 -Xcudafe "--diag_suppress=code_is_unreachable --diag_suppress=initialization_not_reachable" #nvcc gpu compute capability
#NVCCFLAGS= -std=c++11 -Xcompiler "-Mfma -fastsse" #flags for NVCC
NVCCFLAGS= -std=c++11 -Xcompiler "-O3 -mfma -mavx -fabi-version=6" #flags for NVCC
endif
##for use in power 8 nodes
#ifeq ($(strip $(shell domainname)),psg.cluster.zone)
#INCLUDE = -I$(HOME)/include
#CC=pgc++
#MPICC=mpic++
#OPT=-O3
##MPICFLAGS+= -DMPICH_IGNORE_CXX_SEEK
##OMPFLAG=-fopenmp# flag for CC and MPICC
##JSONLIB=-L$(HOME)/include/json/../../src/lib_json -ljsoncpp # json library for input parameters
#NVCCARCH=-arch sm_60 -Xcudafe "--diag_suppress=code_is_unreachable --diag_suppress=initialization_not_reachable" #nvcc gpu compute capability
##NVCCFLAGS= -std=c++11 -Xcompiler "-Mfma -fastsse" #flags for NVCC
#NVCCFLAGS= -std=c++11 -DINSTRSET=7 -Xcompiler "-O3 -Mvect=simd" #flags for NVCC
#endif
#export PATH=$PATH:/usr/local/cuda-9.0/bin
# unset MODULEPATH
# . /etc/profile.d/modules.sh
