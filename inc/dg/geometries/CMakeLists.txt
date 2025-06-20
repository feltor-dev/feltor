# inc/dg/geometries/CMakeLists.txt

# find ../../geometries -name "*.h" -not -name "*doc.h" | sort
# testfunctors.h is not technically included in the library so its changes
# don't technically matter but it does not block anything either
set(GEOMETRIES_HEADERS
  geometries.h
  ../../geometries/adaption.h
  ../../geometries/average.h
  ../../geometries/curvilinear.h
  ../../geometries/curvilinearX.h
  ../../geometries/ds_generator.h
  ../../geometries/ds.h
  ../../geometries/fieldaligned.h
  ../../geometries/fluxfunctions.h
  ../../geometries/flux.h
  ../../geometries/generator.h
  ../../geometries/generatorX.h
  ../../geometries/geometries.h
  ../../geometries/guenter.h
  ../../geometries/hector.h
  ../../geometries/magnetic_field.h
  ../../geometries/make_field.h
  ../../geometries/modified.h
  ../../geometries/mpi_curvilinear.h
  ../../geometries/mpi_fieldaligned.h
  ../../geometries/polar.h
  ../../geometries/polynomial.h
  ../../geometries/polynomial_parameters.h
  ../../geometries/refined_curvilinearX.h
  ../../geometries/ribeiro.h
  ../../geometries/ribeiroX.h
  ../../geometries/separatrix_orthogonal.h
  ../../geometries/sheath.h
  ../../geometries/simple_orthogonal.h
  ../../geometries/solovev.h
  ../../geometries/solovev_parameters.h
  ../../geometries/taylor.h
  ../../geometries/testfunctors.h
  ../../geometries/toroidal.h
  ../../geometries/utilities.h
  ../../geometries/utilitiesX.h
)

# Not setup for catch2 yet
# Some can be compiled for MPI
# find ../../geometries -name "*t.cpp" | sort
set(GEOMETRIES_TESTS
  ../../geometries/average_t.cpp
  ../../geometries/ds_curv_t.cpp
  ../../geometries/ds_guenter_t.cpp
  ../../geometries/ds_straight_t.cpp
  ../../geometries/ds_t.cpp
  ../../geometries/elliptic3d_t.cpp
  ../../geometries/flux_t.cpp
  ../../geometries/modified_t.cpp
  ../../geometries/ribeiroX_t.cpp
  ../../geometries/separatrix_orthogonal_t.cpp
  ../../geometries/solovev_t.cpp
)
# Manually set (all programs with WITH_MPI)
set(GEOMETRIES_MPI_TESTS
  ../../geometries/ds_curv_t.cpp
  ../../geometries/ds_guenter_t.cpp
  ../../geometries/ds_straight_t.cpp
  ../../geometries/ds_t.cpp
)

# find ../../geometries -name "*b.cpp" | sort
set(GEOMETRIES_BENCHMARKS
  ../../geometries/conformal_elliptic_b.cpp
  ../../geometries/conformalX_elliptic_b.cpp
  ../../geometries/geometry_advection_b.cpp
  ../../geometries/geometry_elliptic_b.cpp
  ../../geometries/geometryX_elliptic_b.cpp
  ../../geometries/geometryX_refined_elliptic_b.cpp
)
set(GEOMETRIES_MPI_BENCHMARKS
  ../../geometries/geometry_advection_b.cpp
  ../../geometries/geometry_elliptic_b.cpp
)

# These targets need to link to netcdf
# Test grep "file.h" * in geometries
set( GEOMETRIES_BENCHMARKS_WITH_FILE
  ../../geometries/average_t.cpp
  ../../geometries/flux_t.cpp
  ../../geometries/geometry_elliptic_b.cpp
  ../../geometries/geometryX_elliptic_b.cpp
  ../../geometries/geometryX_refined_elliptic_b.cpp
  ../../geometries/modified_t.cpp
  ../../geometries/ribeiroX_t.cpp
  ../../geometries/separatrix_orthogonal_t.cpp
)


# ============================================
# Create DG geometries library target

add_library(dg_geometries INTERFACE)
set_property(TARGET dg_geometries PROPERTY EXPORT_NAME geometries)
add_library(feltor::dg::geometries ALIAS dg_geometries)
target_sources( dg_geometries INTERFACE
    FILE_SET dg_geometries_headers TYPE HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/inc
    FILES ${GEOMETRIES_HEADERS}
)
target_link_libraries(dg_geometries INTERFACE dg_dg)
target_link_libraries(dg_geometries INTERFACE dg_file_json)

# ============================================
# Add geometries tests
# TODO Disabled for now, tests are not using Catch2 (added to benchmarks instead)

# if(FELTOR_BUILD_TESTS)
#   include(FeltorTests)
#   foreach(test_path IN LISTS GEOMETRIES_TESTS)
#     add_geometries_test("${test_path}")
#   endforeach()
# endif()

# ============================================
# Add geometries benchmarks

if(FELTOR_BUILD_BENCHMARKS)
    include(FeltorBenchmarks)
    if( FELTOR_WITH_MPI)
        foreach(benchmark_path IN LISTS GEOMETRIES_MPI_BENCHMARKS GEOMETRIES_MPI_TESTS)
            if( (${benchmark_path} IN_LIST GEOMETRIES_BENCHMARKS_WITH_FILE) AND FELTOR_FILE_WITH_NETCDF)
                add_dg_benchmark("${benchmark_path}" "geometries" ON target_name)
                target_link_libraries( ${target_name} PRIVATE dg_geometries)
                target_link_libraries( ${target_name} PRIVATE dg_file)
            elseif( NOT ("${benchmark_path}" IN_LIST GEOMETRIES_BENCHMARKS_WITH_FILE) )
                add_dg_benchmark("${benchmark_path}" "geometries" ON target_name)
                target_link_libraries( ${target_name} PRIVATE dg_geometries)
            endif()
        endforeach()
    else()
        foreach(benchmark_path IN LISTS GEOMETRIES_BENCHMARKS GEOMETRIES_TESTS)
            if( (${benchmark_path} IN_LIST GEOMETRIES_BENCHMARKS_WITH_FILE) AND FELTOR_FILE_WITH_NETCDF)
                add_dg_benchmark("${benchmark_path}" "geometries" OFF target_name)
                target_link_libraries( ${target_name} PRIVATE dg_geometries)
                target_link_libraries( ${target_name} PRIVATE dg_file)
            elseif( NOT (${benchmark_path} IN_LIST GEOMETRIES_BENCHMARKS_WITH_FILE) )
                add_dg_benchmark("${benchmark_path}" "geometries" OFF target_name)
                target_link_libraries( ${target_name} PRIVATE dg_geometries)
            endif()
        endforeach()
    endif()
endif()
# ============================================
# Add Install rules

if(FELTOR_INSTALL)
    install(
        TARGETS dg_geometries COMPONENT feltor_dg
        EXPORT feltor_dg
        FILE_SET dg_geometries_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()
