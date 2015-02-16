#usage:
#mkdir build
#cd build
#cmake ../
#make [VERBOSE=1]
#########################################################
cmake_minimum_required( VERSION 2.8) # require minimum cmake version
project( Interactive)  # sets project variables
###############Add executable###################
add_executable( interactive interactive.cpp) # adds dependencies to interactive target
add_executable( opencv_t opencv_t.cpp) # adds dependencies to interactive target
##############FIND LIBRARIES###################
find_package( OpenCV REQUIRED)  # find well-known library for which cmake has config files for many platforms
find_library( FFTW3 fftw3) # find system library
find_package( PkgConfig REQUIRED)
pkg_search_module( GLFW REQUIRED glfw3) # glfw has no cmake config file
find_package( OpenMP REQUIRED )

##############Add include directories###################
include_directories( ${GLFW_INCLUDE_DIRS})
include_directories( /home/matthias/include)
include_directories( $(interactive_SOURCE_DIR)../../inc )


##############Add libs###################
target_link_libraries( opencv_t ${OpenCV_LIBS})
target_link_libraries( interactive ${OpenCV_LIBS})
target_link_libraries( interactive ${GLFW_STATIC_LIBRARIES})
target_link_libraries( interactive ${FFTW3})
#############Set optimization flags#################
#set is used to set environment variables 
set(CMAKE_BUILD_TYPE Release) # sets -O3 -DNDEBUG flags, can also be Debug
#set(CMAKE_BUILD_TYPE Debug) # sets -O3 -DNDEBUG flags, can also be Debug
set(CMAKE_CXX_FLAGS "-std=c++0x")
###############Set compiler flags for OpenMP support##############
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

