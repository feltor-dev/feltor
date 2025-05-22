# inc/dg/matrix/CMakeLists.txt

# find ../../matrix -name "*.h" -not -name "*doc.h" | sort
# testfunctors.h is not technically included in the library so its changes
# don't technically matter but it does not block anything either
set(MATRIX_HEADERS
  matrix.h
  ../../matrix/exp_runge_kutta.h
  ../../matrix/functors.h
  ../../matrix/lanczos.h
  ../../matrix/matrixfunction.h
  ../../matrix/matrix.h
  ../../matrix/matrixsqrt.h
  ../../matrix/mcg.h
  ../../matrix/polarization.h
  ../../matrix/polarization_init.h
  ../../matrix/sqrt_cauchy.h
  ../../matrix/sqrt_ode.h
  ../../matrix/tableau.h
  ../../matrix/tensorelliptic.h
  ../../matrix/tridiaginv.h
)

# Not setup for catch2 yet
# Some can be compiled for MPI
# find ../../matrix -name "*t.cpp" | sort
set(MATRIX_TESTS
)

# find ../../matrix -name "*b.cpp" | sort
set(MATRIX_BENCHMARKS
)


# ============================================
# Create DG matrix library target

# Requires Boost.
# Using Boost::boost in place of Boost::math, as the latter requires linking
# against precompiled Boost libraries, which are not always available.

# From https://github.com/cpm-cmake/CPM.cmake/blob/master/examples/boost/CMakeLists.txt
CPMAddPackage(
  NAME Boost
  VERSION 1.86.0 # Versions less than 1.85.0 may need patches for installation targets.
  URL https://github.com/boostorg/boost/releases/download/boost-1.86.0/boost-1.86.0-cmake.tar.xz
  URL_HASH SHA256=2c5ec5edcdff47ff55e27ed9560b0a0b94b07bd07ed9928b476150e16b0efc57
  OPTIONS "BOOST_ENABLE_CMAKE ON" "BOOST_SKIP_INSTALL_RULES ON" # Set `OFF` for installation
          "BUILD_SHARED_LIBS OFF" #"BOOST_INCLUDE_LIBRARIES container\\\;asio" # Note the escapes!
)

if(NOT Boost_FOUND)
  message(WARNING "Boost not found. The dg::matrix target will not be created.")
  return()
endif()

message(STATUS "Boost version: ${Boost_VERSION}")

add_library(matrix INTERFACE ${MATRIX_HEADERS})
add_library(dg::matrix ALIAS matrix)
target_link_libraries(matrix INTERFACE dg::dg)
target_link_libraries(matrix INTERFACE Boost::boost) # Boost::boost is header only version

# Requires lapacke.
# https://stackoverflow.com/questions/75835663/how-to-link-lapacke-to-a-c-project-using-cmake
include( LAPACKE.cmake)
find_package( LAPACK REQUIRED)
find_package( LAPACKE REQUIRED)
if(NOT LAPACK_FOUND)
  message(WARNING "LAPACK not found. The dg::matrix target will not be created.")
  return()
endif()
target_link_libraries(matrix INTERFACE LAPACKE)


# ============================================
# Add matrix tests
# TODO Disabled for now, tests are not using Catch2

# if(FELTOR_BUILD_TESTS)
#   include(FeltorTests)
#   foreach(test_path IN LISTS GEOMETRIES_TESTS)
#     add_matrix_test("${test_path}")
#   endforeach()
# endif()

# ============================================
# Add matrix benchmarks
# TODO Disabled for now, benchmark are using cu file endings

# if(FELTOR_BUILD_BENCHMARKS)
#   include(FeltorBenchmarks)
#   foreach(benchmark_path IN LISTS GEOMETRIES_BENCHMARKS)
#     add_dg_matrix_benchmark("${benchmark_path}")
#   endforeach()
# endif()
