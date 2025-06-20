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
  ../../matrix/exp_runge_kutta_t.cu
  ../../matrix/functors_t.cu
  ../../matrix/invtridiag_t.cu
  ../../matrix/polarization_init_t.cu
)

# find ../../matrix -name "*b.cpp" | sort
set(MATRIX_BENCHMARKS
  ../../matrix/lanczos_b.cu
  #../../matrix/lanczos_mpib.cu
  ../../matrix/matrixfunction_b.cu
  ../../matrix/matrixsqrt_b.cu
  ../../matrix/polarization_b.cu
  ../../matrix/tensorelliptic2d_b.cu
  ../../matrix/tridiaginv_b.cu
)
# These targets need to link to netcdf
# Test grep "file.h" * in matrix
set( MATRIX_BENCHMARKS_WITH_FILE
  ../../matrix/polarization_b.cu
  ../../matrix/polarization_init_t.cu
)


# ============================================
# Create DG matrix library target

add_library(dg_matrix INTERFACE)
set_property(TARGET dg_geometries PROPERTY EXPORT_NAME geometries)
add_library(feltor::dg::matrix ALIAS dg_matrix)
target_sources( dg_matrix INTERFACE
    FILE_SET dg_matrix_headers TYPE HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/inc
    FILES ${MATRIX_HEADERS}
)
target_link_libraries(dg_matrix INTERFACE dg_dg)


# Requires Boost. Install libboost-dev
# Let's not use CPM here as Boost recommends using find_package and we can
# expect Boost headers to be installed globally on a system
# This also avoids problems if another project uses feltor and includes boost
# before feltor. In such a case CPM complains that Boost::headers target is
# already defined

# # From https://github.com/cpm-cmake/CPM.cmake/blob/master/examples/boost/CMakeLists.txt
# CPMAddPackage(
#   NAME Boost
#   VERSION 1.86.0 # Versions less than 1.85.0 may need patches for installation targets.
#   URL https://github.com/boostorg/boost/releases/download/boost-1.86.0/boost-1.86.0-cmake.tar.xz
#   URL_HASH SHA256=2c5ec5edcdff47ff55e27ed9560b0a0b94b07bd07ed9928b476150e16b0efc57
#   OPTIONS "BOOST_ENABLE_CMAKE ON" "BOOST_SKIP_INSTALL_RULES ON" # Set `OFF` for installation
#           "BUILD_SHARED_LIBS OFF" "BOOST_INCLUDE_LIBRARIES math"
# )
find_package(Boost REQUIRED)
target_link_libraries(dg_matrix INTERFACE Boost::headers) # We use header only version

# Requires LAPACK. Install liblapack-dev
find_package(LAPACK REQUIRED)
target_link_libraries(dg_matrix INTERFACE ${LAPACK_LIBRARIES})


# ============================================
# Add matrix tests
# TODO Disabled for now, tests are not using Catch2 so compiled as benchmarks

# if(FELTOR_BUILD_TESTS)
#   include(FeltorTests)
#   foreach(test_path IN LISTS MATRIX_TESTS)
#     add_dg_test("${test_path}")
#   endforeach()
# endif()

# ============================================
# Add matrix benchmarks

if(FELTOR_BUILD_BENCHMARKS)
  include(FeltorBenchmarks)
  if( FELTOR_WITH_MPI)
  else()
    foreach(benchmark_path IN LISTS MATRIX_BENCHMARKS MATRIX_TESTS)
        if( "${benchmark_path}" IN_LIST MATRIX_BENCHMARKS_WITH_FILE AND FELTOR_FILE_WITH_NETCDF)
            add_dg_benchmark("${benchmark_path}" "matrix" OFF target_name)
            target_link_libraries( ${target_name} PRIVATE dg_matrix)
            target_link_libraries( ${target_name} PRIVATE dg_file)
        elseif( NOT "${benchmark_path}" IN_LIST MATRIX_BENCHMARKS_WITH_FILE )
            add_dg_benchmark("${benchmark_path}" "matrix" OFF target_name)
            target_link_libraries( ${target_name} PRIVATE dg_matrix)
        endif()
    endforeach()
  endif()
endif()
# ============================================
# Add Install rules

if(FELTOR_INSTALL)
    install(
        TARGETS dg_matrix COMPONENT feltor_dg
        EXPORT feltor_dg
        FILE_SET dg_matrix_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()
