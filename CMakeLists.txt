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
## -DFELTOR_FETCH_THRUST=ON, -DFELTOR_FETCH_CUSP=ON, -DFELTOR_FETCH_VCL=ON,
## and -DFELTOR_FETCH_DRAW=ON may also be set individually.
##
## After configuring the project, build it using:
##
##   cmake --build build
##
## This will compile all targets in the project, including all benchmarks and
## tests. A smaller subset of targets can be built using the `--target` flag:
##
##   cmake --build build --target dg_benchmarks  # Build all dg benchmarks
##   cmake --build build --target dg_tests       # Build all dg tests
##   cmake --build build --target dg_blas_b      # Build only inc/dg/blas_b.cu
##   cmake --build build --target dg_blas_t      # Build only inc/dg/blas_t.cu
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
##   ctest --test-dir=build -R dg_blas_t

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

option(FELTOR_BUILD_TESTS "Build tests." ${PROJECT_IS_TOP_LEVEL})
option(FELTOR_BUILD_BENCHMARKS "Build benchmarks." ${PROJECT_IS_TOP_LEVEL})
option(FELTOR_BUILD_DRAW_TESTS "Build tests that require visual inspection.")

option(FELTOR_FETCH_THRUST "Fetch Thrust library from GitHub.")
option(FELTOR_FETCH_CUSP "Fetch CUSP library from GitHub.")
option(FELTOR_FETCH_VCL "Fetch Vector Class (version 1) library from GitHub.")
option(FELTOR_FETCH_DRAW "Fetch DRAW from GitHub.")
option(FELTOR_FETCH_DEPS "Fetch dependencies from GitHub.")

if(FELTOR_USE_OMP AND FELTOR_USE_GPU)
  message(FATAL_ERROR "Cannot set both FELTOR_USE_OMP and FELTOR_USE_GPU")
endif()

if(FELTOR_FETCH_DEPS)
  # Toggle all FELTOR_FETCH_* options ON
  # Note: Overrides -DFELTOR_FETCH_THRUST=OFF, etc.
  set(FELTOR_FETCH_THRUST ON CACHE BOOL "Fetch Thrust library from GitHub." FORCE)
  set(FELTOR_FETCH_CUSP ON CACHE BOOL "Fetch CUSP library from GitHub." FORCE)
  set(FELTOR_FETCH_VCL ON CACHE BOOL "Fetch Vector Class (version 1) library from GitHub." FORCE)
  set(FELTOR_FETCH_DRAW ON CACHE BOOL "Fetch DRAW from GitHub." FORCE)
endif()

if(FELTOR_USE_GPU)
  enable_language(CUDA)
endif()

# ============================================
# Set up compiler-specific flags

if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(FELTOR_FLAGS_CXX 
    "-Wall"
    "-Wextra"
    "-Wpedantic"
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
  "-Xcompiler=\"${FELTOR_FLAGS_CXX}\""
  "--extended-lambda"
)

# ============================================
# Create DG library target

# Bring dg variables into scope
add_subdirectory("${PROJECT_SOURCE_DIR}/inc/dg")

# Create the dg target library.
# As a header-only library, this must be an interface library.
# This means it has no sources, but can link/be linked to other
# libraries, have dependencies, etc.
add_library(dg INTERFACE ${DG_HEADERS})
# Ensure that `#include "dg/..."` works
target_include_directories(dg INTERFACE "${PROJECT_SOURCE_DIR}/inc")
# Set the minimum C++ standard
target_compile_features(dg INTERFACE cxx_std_14)
# Set flags, whether using C++ or CUDA
target_compile_options(dg INTERFACE 
  $<$<COMPILE_LANGUAGE:CXX>:${FELTOR_FLAGS_CXX}>
  $<$<COMPILE_LANGUAGE:CUDA>:${FELTOR_FLAGS_CUDA}>
)

# Create an alias: used to link to the library from external projects.
add_library(feltor::dg ALIAS dg)

# ============================================
# Add dg dependencies
include(FeltorFetchDeps)

# We're pinned to a specific version of Thrust for CUSP compatibility.
# This doesn't include Thrust's modern CMakes support, so we're making
# use of the FindThrust.cmake module.
if(FELTOR_FETCH_THRUST)
  fetch_thrust()
else()
  find_package(Thrust REQUIRED 1.9.3)
endif()
target_link_libraries(dg INTERFACE thrust::thrust)
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

if(FELTOR_FETCH_CUSP)
  fetch_cusp()
else()
  find_package(CUSP REQUIRED)
endif()
target_link_libraries(dg INTERFACE cusp::cusp)

if(FELTOR_USE_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(dg INTERFACE MPI::MPI_CXX)
endif()

if(FELTOR_USE_OMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(dg INTERFACE OpenMP::OpenMP_CXX)
endif()

# ============================================
# Set up optional DRAW dependency

if(FELTOR_FETCH_DRAW)
  fetch_draw()
else()
  find_package(DRAW)
endif()

if(DRAW_FOUND)
  # Link to GLFW and OpenGL
  # Must be installed separately, e.g. via `sudo apt install libglfw3-dev`
  find_package(glfw3)
  if(glfw3_FOUND)
    target_link_libraries(draw::draw INTERFACE glfw)
  else()
    message(WARNING "Could not find GLFW3, which is needed for DRAW.")
  endif()
  find_package(OpenGL)
  if(OpenGL_FOUND)
    target_link_libraries(draw::draw INTERFACE OpenGL::GL)
  else()
    message(WARNING "Could not find OpenGL, which is needed for DRAW.")
  endif()
endif()

# ============================================
# Add dg benchmarks

if(FELTOR_BUILD_BENCHMARKS)
  include(FeltorBenchmarks)
  add_custom_target(dg_benchmarks)

  foreach(benchmark_path IN LISTS DG_BENCHMARKS)
    add_dg_benchmark("${benchmark_path}")
  endforeach()

  if(FELTOR_USE_MPI)
    foreach(benchmark_path IN LISTS DG_MPI_BENCHMARKS)
      add_dg_benchmark("${benchmark_path}")
    endforeach()
  endif()
endif()

# ============================================
# Add dg tests

# TODO Tests should be written to simply pass or fail, ideally using
# a framework like GoogleTest. When running with CTest, some of these
# tests hang as they require user input.

if(FELTOR_BUILD_TESTS)
  include(CTest)
  include(FeltorTests)
  enable_testing()
  add_custom_target(dg_tests)

  foreach(test_path IN LISTS DG_TESTS)
    add_dg_test("${test_path}")
  endforeach()

  if(FELTOR_USE_MPI)
    foreach(test_path IN LISTS DG_MPI_TESTS)
      add_dg_test("${test_path}")
    endforeach()
  endif()
endif()

# Draw tests require visual inspection, so they are built separately
# and not added to the dg_tests target, nor are they run by CTest.
if(FELTOR_BUILD_DRAW_TESTS)
  if(NOT DRAW_FOUND)
    message(FATAL_ERROR "DRAW not found. Cannot build DRAW tests.")
  endif()
  add_custom_target(draw_tests)
  foreach(test_path IN LISTS DG_DRAW_TESTS)
    add_draw_test("${test_path}")
  endforeach()
endif()