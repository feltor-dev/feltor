cmake_minimum_required(VERSION 3.20)

# Project name and version
project(feltor VERSION 1.0 LANGUAGES CXX)

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
option(FELTOR_BUILD_TESTS "Build tests.")
option(FELTOR_BUILD_BENCHMARKS "Build benchmarks.")

if(FELTOR_USE_OMP AND FELTOR_USE_GPU)
  message(FATAL_ERROR "Cannot set both FELTOR_USE_OMP and FELTOR_USE_GPU")
endif()

if(FELTOR_USE_GPU)
  enable_language(CUDA)
endif()

# ============================================
# Create DG library target

# Bring dg variables into scope
add_subdirectory("${PROJECT_SOURCE_DIR}/inc/dg")

# Create the dg target library.
# As a header-only library, this must be an interface library.
# This means it has no sources, but can link/be linked to other
# libraries, have dependencies, etc.
add_library(dg INTERFACE ${DG_CXX_HEADERS})
if(FELTOR_USE_GPU)
  target_sources(dg INTERFACE ${DG_CU_HEADERS})
endif()
# Ensure that `#include "dg/..."` works
target_include_directories(dg INTERFACE "${PROJECT_SOURCE_DIR}/inc")
# Set the minimum C++ standard
target_compile_features(dg INTERFACE cxx_std_14)

# Create an alias: This can be used to link to library from external projects.
add_library(feltor::dg ALIAS dg)

# ============================================
# Add dg dependencies

# We're pinned to a specific version of Thrust for CUSP compatibility.
# This doesn't include Thrust's modern CMakes support, so we're making
# use of the FindThrust.cmake module.
find_package(Thrust REQUIRED 1.9.3)
target_link_libraries(dg INTERFACE thrust::thrust)
if(FELTOR_USE_OMP)
  target_compile_definitions(dg INTERFACE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)
elseif(FELTOR_USE_GPU)
  target_compile_definitions(dg INTERFACE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)
else()
  target_compile_definitions(dg INTERFACE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
endif()

find_package(VCL REQUIRED)
target_link_libraries(dg INTERFACE vcl::vcl)

find_package(CUSP REQUIRED)
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
# Add dg benchmarks

if(FELTOR_BUILD_BENCHMARKS)
  add_custom_target(dg_benchmarks)

  function(add_dg_benchmark benchmark_path)
    # Get a unique name for the benchmark
    cmake_path(GET benchmark_path STEM benchmark_stem)
    set(benchmark_name "dg_${benchmark_stem}")
    # Optionally treat .cu files as .cpp files
    if(NOT FELTOR_USE_GPU)
      set_source_files_properties(${benchmark_path} PROPERTIES
        LANGUAGE CXX
        COMPILE_FLAGS "-x c++"
      )
    endif()
    # Create the benchmark executable and link dependencies
    add_executable(${benchmark_name} "${benchmark_path}")
    target_link_libraries(${benchmark_name} PRIVATE feltor::dg)
    # Set compile warnings
    target_compile_options(${benchmark_name} PRIVATE -Wall -Wextra -pedantic)
    # Add as dependency to the dg_benchmarks target
    add_dependencies(dg_benchmarks ${benchmark_name})
    # Ensure built benchmarks are built in their own directory
    set_target_properties(${benchmark_name} PROPERTIES 
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks"
    )
  endfunction()

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

# TODO Add DG_DRAW_TESTS. These shouldn't be run via CTest, as they need
# to be visually inspected.

if(FELTOR_BUILD_TESTS)
  include(CTest)
  enable_testing()
  add_custom_target(dg_tests)

  function(add_dg_test test_path)
    # Get a unique name for the test
    cmake_path(GET test_path STEM test_stem)
    set(test_name "dg_${test_stem}")
    # Optionally treat .cu files as .cpp files
    if(NOT FELTOR_USE_GPU)
      set_source_files_properties(${test_path} PROPERTIES
        LANGUAGE CXX
        COMPILE_FLAGS "-x c++"
      )
    endif()
    # Create the test executable and link dependencies
    add_executable(${test_name} "${test_path}")
    target_link_libraries(${test_name} PRIVATE feltor::dg)
    # Set compile warnings
    target_compile_options(${test_name} PRIVATE -Wall -Wextra -pedantic)
    # Add as dependency to the dg_tests target
    add_dependencies(dg_tests ${test_name})
    # Ensure built tests are built in their own directory
    set_target_properties(${test_name} PROPERTIES 
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
    # Register the test with CTest
    add_test(NAME ${test_name} COMMAND "tests/${test_name}")
  endfunction()

  foreach(test_path IN LISTS DG_TESTS)
    add_dg_test("${test_path}")
  endforeach()

  if(FELTOR_USE_MPI)
    foreach(test_path IN LISTS DG_MPI_TESTS)
      add_dg_test("${test_path}")
    endforeach()
  endif()
endif()