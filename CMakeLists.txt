cmake_minimum_required(VERSION 3.26)

# Project name and version
project(feltor VERSION 8.2.0 LANGUAGES CXX)

## Usage:
##
## Configure the project using:
##
##   cmake --preset cpu ## or gpu or omp or mpi-gpu or mpi-cpu or mpi-omp
##
## This will configure the project to build in the `build` directory, and will
## build a plain C++ project. It can be configured to run as a single-threaded
## CPU program by setting -DCCCL_THRUST_DEVICE_SYSTEM=CPP, a multi-threaded CPU
## program using -DCCCL_THRUST_DEVICE_SYSTEM=OMP, or as a CUDA-enabled GPU program
## with -DCCCL_THRUST_DEVICE_SYSTEM=CUDA. By default, the project will be built
## for GPU.
##
## To enable MPI, set the option -DFELTOR_WITH_MPI=ON.
##
## The location of dependencies can be specified with -DCMAKE_PREFIX_PATH. To
## automatically fetch dependencies from GitHub, set -DFELTOR_FETCH_DEPS=ON.
## -DFELTOR_FETCH_CCCL=ON, -DFELTOR_FETCH_VCL=ON, and -DFELTOR_FETCH_DRAW=ON
## may also be set individually.
##
## When building the FELTOR executable, the user may set -DFELTOR_WITH_GLFW=ON
## to enable plotting support.
##
## After configuring the project, build it using:
##
##   cmake --build build
##
## This will compile all targets in the project, including all benchmarks and
## tests. A smaller subset of targets can be built using the `--target` flag:
##
##   cmake --build build --target feltor         # Build the feltor executable
##   cmake --build build --target dg_benchmarks  # Build all dg benchmarks
##   cmake --build build --target dg_tests       # Build all dg tests
##   cmake --build build --target dg_blas_b      # Build only inc/dg/blas_b.cpp
##   cmake --build build --target dg_blas_t      # Build only inc/dg/blas_t.cpp
##
## Compilation can be sped up by supplying the `--parallel` flag, while the
## `--clean-first` flag will force a clean build. Compiled benchmarks may be
## found in `build/benchmarks`, and compiled tests can be found in
## `build/tests`. To run the tests, use the `ctest` command:
##
##   ctest --test-dir=build
##
## Individual tests may be run using:
##
##   ctest --test-dir=build -R blas
##
## The test names are defined within the test files themselves.

# Enable support for IDEs like Visual Studio
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Add cmake modules (e.g. FindThrust.cmake)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# ============================================
# User Options

# Set these using -DFELTOR_WITH_MPI=ON, etc.
# By default, compiles single threaded on a CPU.

option(FELTOR_WITH_MPI "Switch on multiprocessing with MPI.")
# TODO Shouldn't the file library expose this option?
option(FELTOR_USE_JSONCPP "Use jsoncpp instead of nlohmann/json.")

option(FELTOR_BUILD_EXE "Build Feltor executable." ${PROJECT_IS_TOP_LEVEL})
option(FELTOR_BUILD_TESTS "Build tests." ${PROJECT_IS_TOP_LEVEL})
option(FELTOR_BUILD_BENCHMARKS "Build benchmarks." ${PROJECT_IS_TOP_LEVEL})

option(FELTOR_WITH_GLFW "Build with plotting support")

# ============================================
# Make cpm package manager visible here and in subdirectories
include( CPM)

# ============================================
# Set up testing dependencies

if(FELTOR_BUILD_TESTS)
  enable_testing()
  CPMAddPackage(
    NAME Catch2
    GITHUB_REPOSITORY catchorg/Catch2
    VERSION 3.8.0
  )
  # Make include(Catch2) available according to
  # https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md
  list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
endif()

# ============================================
# Set up compiler-specific flags

if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # Currently raises too many warnings!
  set(FELTOR_FLAGS_CXX
    "-Wall"
    "-Wextra"
    "-Wpedantic"
    #"-mfma"
    #"-mavx"
    #"-march=native"
  )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(FELTOR_FLAGS_CXX
    "/W4"
    #"/arch:AVX2"
  )
else()
  set(FELTOR_FLAGS_CXX "")
endif()

if(FELTOR_WITH_MPI)
  find_package(MPI REQUIRED)
  # target_link_libraries(dg INTERFACE MPI::MPI_CXX)
endif()


# =============================================
# Set up DG header-only library
# Defines the targets:
# - dg library, alias dg::dg
# - dg file library, alias dg::file
# - dg_benchmarks
# - dg_tests

# This is a somewhat chicken-egg problem: The cuda language can only be enabled
# once in the project. But before including the cccl package the device macro
# is not defined. So we must assume the user defines CCCL_THRUST_DEVICE_SYSTEM
# and/or if they do not, then the default is also CUDA
if(CCCL_THRUST_DEVICE_SYSTEM STREQUAL "CUDA" OR CCCL_THRUST_DEVICE_SYSTEM STREQUAL "")
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()
add_subdirectory("${PROJECT_SOURCE_DIR}/inc/dg")
#set(FELTOR_FLAGS_CUDA
#  "-arch=sm_61"
#  "--extended-lambda"
#)



# ============================================
# Add FELTOR executable
# TODO Add all other executables

if(FELTOR_BUILD_EXE)
  add_subdirectory("${PROJECT_SOURCE_DIR}/src/feltor")
endif()

