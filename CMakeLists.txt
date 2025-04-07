cmake_minimum_required(VERSION 3.21)

# Project name and version
project(feltor VERSION 1.0 LANGUAGES CXX)

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
option(FELTOR_FETCH_VCL "Fetch Vector Class (version 1) library from GitHub.")
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
  set(FELTOR_FETCH_VCL ON CACHE BOOL "Fetch Vector Class (version 1) library from GitHub." FORCE)
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

# ============================================
# Create DG library target

# Bring dg variables into scope
add_subdirectory("${PROJECT_SOURCE_DIR}/inc/dg")

# Create the dg target library.
# As a header-only library, this must be an interface library.
# This means it has no sources, but can link/be linked to other
# libraries, have dependencies, etc. These properties are then
# inherited by any target that links to it.
add_library(dg INTERFACE ${DG_HEADERS})
# Ensure that `#include "dg/..."` works
target_include_directories(dg INTERFACE "${PROJECT_SOURCE_DIR}/inc")
# Set the minimum C++ standard
target_compile_features(dg INTERFACE cxx_std_17)
# Set flags
# CMake switches on -forward-unknown-to-host-compiler by default, so C++ flags
# may be set even when compiling for CUDA.
target_compile_options(dg INTERFACE
  "${FELTOR_FLAGS_CXX}"
  $<$<COMPILE_LANGUAGE:CUDA>:${FELTOR_FLAGS_CUDA}>
)

# Create an alias: used to link to the library from external projects.
add_library(feltor::dg ALIAS dg)

# ============================================
# Add dg dependencies
include(FeltorFetchDeps)

if(FELTOR_FETCH_CCCL)
  fetch_cccl()
else()
  find_package(cccl REQUIRED)
endif()
target_link_libraries(dg INTERFACE Thrust::Thrust)
target_link_libraries(dg INTERFACE CUB::CUB)
target_link_libraries(dg INTERFACE libcudacxx::libcudacxx)
if(FELTOR_USE_OMP)
  target_compile_definitions(dg INTERFACE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)
elseif(FELTOR_USE_GPU)
  target_compile_definitions(dg INTERFACE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)
else()
  target_compile_definitions(dg INTERFACE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
endif()

if(FELTOR_FETCH_VCL)
  fetch_vcl()
else()
  find_package(VCL REQUIRED)
endif()
target_link_libraries(dg INTERFACE vcl::vcl)

if(FELTOR_USE_MPI)
  find_package(MPI REQUIRED)
endif()

if(FELTOR_USE_OMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(dg INTERFACE OpenMP::OpenMP_CXX)
endif()

if(FELTOR_USE_JSONCPP)
  # Must install separately, e.g. `sudo apt install libjsoncpp-dev`
  find_package(jsoncpp REQUIRED)
  target_link_libraries(dg INTERFACE jsoncpp_lib)
else()
  if(FELTOR_FETCH_JSON)
    fetch_json()
  else()
    find_package(nlohmann_json REQUIRED)
  endif()
  target_link_libraries(dg INTERFACE nlohmann_json::nlohmann_json)
  target_compile_definitions(dg INTERFACE DG_USE_JSONHPP)
endif()

# Must install separately, e.g. `sudo apt install libnetcdf-dev`
find_package(netCDF REQUIRED)
target_link_libraries(dg INTERFACE netCDF::netcdf)

# ============================================
# Set up optional DRAW dependency

if(FELTOR_WITH_GLFW)
  if(FELTOR_FETCH_DRAW)
    fetch_draw()
  else()
    find_package(DRAW REQUIRED)
  endif()

  # Link to GLFW and OpenGL
  # Must be installed separately, e.g. `sudo apt install libglfw3-dev`
  find_package(glfw3 REQUIRED)
  target_link_libraries(draw::draw INTERFACE glfw)

  find_package(OpenGL REQUIRED)
  target_link_libraries(draw::draw INTERFACE OpenGL::GL)
endif()

# ============================================
# Add FELTOR executable

if(FELTOR_BUILD_EXE)
  add_subdirectory("${PROJECT_SOURCE_DIR}/src")
endif()

# ============================================
# Add dg benchmarks

if(FELTOR_BUILD_BENCHMARKS)
  include(FeltorBenchmarks)
  add_custom_target(dg_benchmarks)

  foreach(benchmark_path IN LISTS DG_BENCHMARKS)
    add_dg_benchmark("${benchmark_path}")
  endforeach()
endif()

# ============================================
# Add dg tests

if(FELTOR_BUILD_TESTS)
  include(CTest)
  include(FeltorTests)
  enable_testing()
  if(FELTOR_FETCH_CATCH2)
    fetch_catch2()
    # Make Catch2 utilties available to include()
    list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
  else()
    find_package(Catch2 3 REQUIRED)
  endif()
  include(Catch)

  add_custom_target(dg_tests)
  foreach(test_path IN LISTS DG_TESTS)
    add_dg_test("${test_path}")
  endforeach()
  if(FELTOR_USE_MPI)
    foreach(test_path IN LISTS DG_MPI_TESTS)
      add_dg_mpi_test("${test_path}")
    endforeach()
  endif()
endif()
