cmake_minimum_required(VERSION 3.26)

# Project name and version
project(feltor VERSION 8.1.1 LANGUAGES CXX)

## Usage:
##
## Configure the project using:
##
##   cmake -Bbuild .
##
## This will configure the project to build in the `build` directory, and will
## build a plain C++ project. To enable MPI, OpenMP, or GPU support, set the
## options -DFELTOR_USE_MPI=ON, -DFELTOR_USE_OMP=ON, or -DFELTOR_USE_GPU=ON.
## Note that OpenMP and GPU support are mutually exclusive.
##
## The location of dependencies can be specified with -DCMAKE_PREFIX_PATH. To
## automatically fetch dependencies from GitHub, set -DFELTOR_FETCH_DEPS=ON.
## -DFELTOR_FETCH_CCCL=ON, -DFELTOR_FETCH_VCL=ON, and -DFELTOR_FETCH_DRAW=ON
## may also be set individually.
##
## When building the FELTOR executable, the user may set -DFELTOR_WITH_GLFW=ON
## to enable plotting support, and -DFELTOR_WITH_NAVIER_STOKES=ON to build with
## the Navier-Stokes extension.
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
## However, some tests currently require user input, and will hang when run.
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

# Set these using -DFELTOR_USE_MPI=ON, etc.
# By default, compiles single threaded on a CPU.

option(FELTOR_USE_MPI "Switch on multiprocessing with MPI.")
option(FELTOR_USE_OMP "Switch on OpenMP multithreading.")
option(FELTOR_USE_GPU "Switch on GPU acceleration.")
option(FELTOR_USE_JSONCPP "Use jsoncpp instead of nlohmann/json.")

option(FELTOR_BUILD_EXE "Build Feltor executable." ${PROJECT_IS_TOP_LEVEL})
option(FELTOR_BUILD_TESTS "Build tests." ${PROJECT_IS_TOP_LEVEL})
option(FELTOR_BUILD_BENCHMARKS "Build benchmarks." ${PROJECT_IS_TOP_LEVEL})

option(FELTOR_WITH_GLFW "Build with plotting support")
option(FELTOR_WITH_NAVIER_STOKES "Build with Navier-Stokes library.")

option(FELTOR_FETCH_CCCL "Fetch NVIDIA's CUDA Core Compute Library (CCCL) from GitHub.")
option(FELTOR_FETCH_VCL "Fetch Vector Class (version 2) library from GitHub.")
option(FELTOR_FETCH_DRAW "Fetch DRAW from GitHub.")
option(FELTOR_FETCH_CATCH2 "Fetch Catch2 from GitHub.")
option(FELTOR_FETCH_JSON "Fetch nlohmann/json from GitHub.")
option(FELTOR_FETCH_DEPS "Fetch dependencies from GitHub.")

if(FELTOR_USE_OMP AND FELTOR_USE_GPU)
  message(FATAL_ERROR "Cannot set both FELTOR_USE_OMP and FELTOR_USE_GPU")
endif()

if(FELTOR_FETCH_DEPS)
  # Toggle all FELTOR_FETCH_* options ON
  # Note: Overrides -DFELTOR_FETCH_CCCL=OFF, etc.
  set(FELTOR_FETCH_CCCL ON CACHE BOOL "Fetch NVIDIA's CUDA Core Compute Library (CCCL) from GitHub." FORCE)
  set(FELTOR_FETCH_VCL ON CACHE BOOL "Fetch Vector Class (version 2) library from GitHub." FORCE)
  set(FELTOR_FETCH_CATCH2 ON CACHE BOOL "Fetch Catch2 from GitHub." FORCE)
  set(FELTOR_FETCH_JSON ON CACHE BOOL "Fetch nlohmann/json from GitHub." FORCE)
  if(FELTOR_WITH_GLFW)
    set(FELTOR_FETCH_DRAW ON CACHE BOOL "Fetch DRAW from GitHub." FORCE)
  else()
    set(FELTOR_FETCH_DRAW OFF CACHE BOOL "Fetch DRAW from GitHub." FORCE)
  endif()
endif()

if(FELTOR_USE_JSONCPP)
  set(FELTOR_FETCH_JSON OFF CACHE BOOL "Fetch nlohmann/json from GitHub." FORCE)
endif()

if(FELTOR_USE_GPU)
  enable_language(CUDA)
endif()

# ============================================
# Set up testing dependencies

if(FELTOR_BUILD_TESTS)
  enable_testing()
  if(FELTOR_FETCH_CATCH2)
    include(FeltorFetchDeps)
    fetch_catch2()
    # Make Catch2 utilties available to include()
    list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
  else()
    find_package(Catch2 3 REQUIRED)
  endif()
endif()

# ============================================
# Set up compiler-specific flags

if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # TODO: Should also set -Wextra and -Wpedantic.
  # Currently raises too many warnings!
  set(FELTOR_FLAGS_CXX
    "-Wall"
    "-mfma"
    "-mavx"
    "-march=native"
  )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(FELTOR_FLAGS_CXX "/W4")
else()
  set(FELTOR_FLAGS_CXX "")
endif()

set(FELTOR_FLAGS_CUDA
  "-arch=sm_61"
  "--extended-lambda"
)

# =============================================
# Set up DG header-only library
# Defines the targets:
# - dg library, alias dg::dg
# - dg file library, alias dg::file
# - dg_benchmarks
# - dg_tests

add_subdirectory("${PROJECT_SOURCE_DIR}/inc/dg")

# ============================================
# Add FELTOR executable

if(FELTOR_BUILD_EXE)
  add_subdirectory("${PROJECT_SOURCE_DIR}/src")
endif()

