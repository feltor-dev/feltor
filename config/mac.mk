ifeq ($(strip $(device)),mac_cpu)
CC=g++
CFLAGS=-Wall -std=c++14
CFLAGS+=-x c++ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP

INCLUDE+= -I/opt/homebrew/Cellar/netcdf/4.8.1_3/include # Include path to netcdf
INCLUDE+= -I/opt/homebrew/Cellar/jsoncpp/1.9.5/include #Include path to jsoncpp
JSONLIB= -L/opt/homebrew/Cellar/jsoncpp/1.9.5/lib -ljsoncpp
LIBS=-L/opt/homebrew/Cellar/netcdf/4.8.1_3/lib -lnetcdf -L/opt/homebrew/Cellar/hdf5/1.12.2_1/lib -lhdf5 -lhdf5_hl
GLFLAGS =-DWITHOUT_GLFW#$$(pkg-config --static --libs glfw3 gl) #-lglfw
endif #device=cpu

##
#For a mac, we need X-code to compile. At the same time, we need to include the paths to the json, hdf5 and netcdf libraries. We work without GLFW, as it gives problems.
# As it is now, the libraries are installed with homebrew, one of the prefered installers for mac. If other programs are used, it is neccessary to include the paths to the libraries used.
# We compile with g++, but other compilers might also work (like clang++).
