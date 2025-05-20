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

# Suppress warning due to new FindBoost policy
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

find_package(Boost 1.5 COMPONENTS math)

if(NOT Boost_FOUND)
  message(WARNING "Boost not found. The dg::matrix target will not be created.")
  return()
endif()

message(STATUS "Boost version: ${Boost_VERSION}")

add_library(matrix INTERFACE ${MATRIX_HEADERS})
add_library(dg::matrix ALIAS matrix)
target_link_libraries(matrix INTERFACE dg::dg)
target_link_libraries(matrix INTERFACE Boost::boost)

# Requires lapacke.
# TODO Not sure how to best setup lapacke
# https://stackoverflow.com/questions/75835663/how-to-link-lapacke-to-a-c-project-using-cmake
find_package( LAPACK REQUIRED)
if(NOT LAPACK_FOUND)
  message(WARNING "LAPACK not found. The dg::matrix target will not be created.")
  return()
endif()
target_link_libraries(matrix INTERFACE LAPACK::LAPACK)


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
