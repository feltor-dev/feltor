# inc/dg/CMakeLists.txt

# find . -type f \( -name "*.h" -or -name "*.cuh" -or -name "*.hpp" \) -not -path "./exblas/*" -not -path "./matrix/*" -not -path "./geometries/*" -not -path "./file/*" -not -name "*doc.h"| sort
# Manually copy result here and replace ./ with 2 whitespaces
set(DG_HEADERS
  adaptive.h
  advection.h
  algorithm.h
  andersonacc.h
  arakawa.h
  backend/blas1_cuda.cuh
  backend/blas1_dispatch_map.h
  backend/blas1_dispatch_mpi.h
  backend/blas1_dispatch_scalar.h
  backend/blas1_dispatch_shared.h
  backend/blas1_dispatch_vector.h
  backend/blas1_omp.h
  backend/blas1_serial.h
  backend/blas2_cusp.h
  backend/blas2_densematrix.h
  backend/blas2_dispatch_map.h
  backend/blas2_dispatch_mpi.h
  backend/blas2_dispatch_scalar.h
  backend/blas2_dispatch_shared.h
  backend/blas2_dispatch_vector.h
  backend/blas2_selfmade.h
  backend/blas2_sparseblockmat.h
  backend/blas2_stencil.h
  backend/config.h
  backend/densematrix.h
  backend/exblas/accumulate.cuh
  backend/exblas/accumulate.h
  backend/exblas/config.h
  backend/exblas/exblas.h
  backend/exblas/exdot_cuda.cuh
  backend/exblas/exdot_omp.h
  backend/exblas/exdot_serial.h
  backend/exblas/ExSUM.FPE.hpp
  backend/exblas/fpedot_cuda.cuh
  backend/exblas/fpedot_omp.h
  backend/exblas/fpedot_serial.h
  backend/exblas/mpi_accumulate.h
  backend/exblas/mylibm.cuh
  backend/exblas/mylibm.hpp
  backend/exceptions.h
  backend/execution_policy.h
  backend/fma.h
  backend/index.h
  backend/matrix_categories.h
  backend/memory.h
  backend/mpi_datatype.h
  backend/mpi_gather.h
  backend/mpi_gather_kron.h
  backend/mpi_init.h
  backend/mpi_kron.h
  backend/mpi_matrix.h
  backend/mpi_permutation.h
  backend/mpi_vector.h
  backend/predicate.h
  backend/scalar_categories.h
  backend/sparseblockmat_cpu_kernels.h
  backend/sparseblockmat_gpu_kernels.cuh
  backend/sparseblockmat.h
  backend/sparseblockmat_omp_kernels.h
  backend/sparsematrix_cpu.h
  backend/sparsematrix_gpu.cuh
  backend/sparsematrix.h
  backend/sparsematrix_omp.h
  backend/tensor_traits_cusp.h
  backend/tensor_traits.h
  backend/tensor_traits_scalar.h
  backend/tensor_traits_std.h
  backend/tensor_traits_thrust.h
  backend/timer.h
  backend/typedefs.h
  backend/vector_categories.h
  backend/view.h
  bicgstabl.h
  blas1.h
  blas2.h
  blas.h
  chebyshev.h
  elliptic.h
  enums.h
  eve.h
  extrapolation.h
  functors.h
  helmholtz.h
  implicit.h
  lgmres.h
  multigrid.h
  multistep.h
  multistep_tableau.h
  nullstelle.h
  ode.h
  pcg.h
  poisson.h
  refined_elliptic.h
  runge_kutta.h
  simpsons.h
  subroutines.h
  tableau.h
  topology/average.h
  topology/base_geometry.h
  topology/base_geometryX.h
  topology/derivativesA.h
  topology/derivatives.h
  topology/derivativesT.h
  topology/derivativesX.h
  topology/dlt.h
  topology/dx.h
  topology/dxX.h
  topology/evaluation.h
  topology/evaluationX.h
  topology/fast_interpolation.h
  topology/fem.h
  topology/fem_weights.h
  topology/filter.h
  topology/functions.h
  topology/geometry.h
  topology/grid.h
  topology/gridX.h
  topology/interpolation.h
  topology/interpolationX.h
  topology/mpi_base_geometry.h
  topology/mpi_derivatives.h
  topology/mpi_evaluation.h
  topology/mpi_grid.h
  topology/mpi_projection.h
  topology/mpi_prolongation.h
  topology/mpi_weights.h
  topology/multiply.h
  topology/operator.h
  topology/operator_tensor.h
  topology/projection.h
  topology/projectionX.h
  topology/prolongation.h
  topology/refined_grid.h
  topology/refined_gridX.h
  topology/split_and_join.h
  topology/stencil.h
  topology/tensor.h
  topology/transform.h
  topology/weights.h
  topology/weightsX.h
  topology/xspacelib.h
)

# These can be compiled for MPI as well
# find . -type f -name "*b.cpp" | sort
set(DG_BENCHMARKS
  arakawa_b.cpp
  blas_b.cpp
  cg2d_b.cpp
  cluster_b.cpp
  elliptic2d_b.cpp
  ellipticX2d_b.cpp
  elliptic_b.cpp
  helmholtz_b.cpp
  helmholtzg2_b.cpp
  multigrid_b.cpp
  poisson_b.cpp
  refined_elliptic2d_b.cpp
)

# These can be compiled for MPI as well
# find . -type f -name "*_t.cpp" | sort
set(DG_TESTS
  adaptive_t.cpp
  arakawa_t.cpp
  backend/fma_t.cpp
  backend/index_t.cpp
  backend/memory_t.cpp
  backend/sparsematrix_t.cpp
  backend/traits_t.cpp
  backend/view_t.cpp
  bicgstabl_t.cpp
  blas1_t.cpp
  blas_t.cpp
  eve_t.cpp
  extrapolation_t.cpp
  helmholtz_t.cpp
  implicit_t.cpp
  multistep_t.cpp
  nullstelle_t.cpp
  pcg_t.cpp
  poisson_t.cpp
  runge_kutta_t.cpp
  simpsons_t.cpp
  topology/average_t.cpp
  topology/base_geometry_t.cpp
  topology/derivatives_t.cpp
  topology/derivativesX_t.cpp
  topology/dlt_t.cpp
  topology/dx_t.cpp
  topology/evaluation_t.cpp
  topology/fem_t.cpp
  topology/filter_t.cpp
  topology/geometry_t.cpp
  topology/grid_t.cpp
  topology/interpolation_t.cpp
  topology/multiply_t.cpp
  topology/operator_t.cpp
  topology/operator_tensor_t.cpp
  topology/projection_t.cpp
  topology/refined_grid_t.cpp
  topology/stencil_t.cpp
  topology/tensor_t.cpp
  topology/transform_t.cpp
)

# These are pure MPI tests
# find . -type f -name "*mpit.cpp" | sort
## TODO some mpi tests are missing (see Makefile MPITargets!)
set(DG_MPI_TESTS
  backend/mpi_exblas_mpit.cpp
  backend/mpi_gather_kron_mpit.cpp
  backend/mpi_gather_mpit.cpp
  backend/mpi_init_mpit.cpp
  backend/mpi_kron_mpit.cpp
  backend/mpi_permutation_mpit.cpp
  topology/filter_mpit.cpp
  topology/interpolation_mpit.cpp
  topology/mpi_grid_mpit.cpp
  topology/stencil_mpit.cpp
)

# Manually list tests that can be compiled for MPI
# Cf Makefile, backend/Makefile and topology/Makefile
list( APPEND DG_MPI_TESTS
    topology/evaluation_t.cpp
    topology/dx_t.cpp
    topology/derivatives_t.cpp
    topology/projection_t.cpp
    arakawa_t.cpp
    poisson_t.cpp
    bicgstabl_t.cpp
    blas_t.cpp
    blas1_t.cpp
)

# ============================================
# Create DG library target

# Create the dg target library.
# As a header-only library, this must be an interface library.
# This means it has no sources, but can link/be linked to other
# libraries, have dependencies, etc. These properties are then
# inherited by any target that links to it.
add_library(dg_dg INTERFACE ${DG_HEADERS})
# When installing the target we prepend the namespace dg::
set_property(TARGET dg_dg PROPERTY EXPORT_NAME dg)
# Create an alias: used to link to the library from external projects.
# consistently with either subdirectory'd and installed library
add_library(feltor::dg::dg ALIAS dg_dg)

# Set the minimum C++ standard
target_compile_features(dg_dg INTERFACE cxx_std_17)

# Ensure that `#include "dg/..."` works
# for both build and install phases (cf "Professional CMake, Ch Install")
target_include_directories(dg_dg INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/inc>
)

# ============================================
# Add dg dependencies

# Through the exblas library we depend on vcl
# VCL does not use cmake so we have to configure manually
CPMAddPackage(
 NAME vcl
 GITHUB_REPOSITORY vectorclass/version2
 GIT_TAG v2.02.01
)

if( vcl_ADDED)
    # with vcl it is recommended to activate and optimize instruction set flags
    message( STATUS "Found Vectorclass in ${vcl_SOURCE_DIR}")
    add_library( vcl::vcl INTERFACE IMPORTED) # imported is visible only in current dir and below
    target_include_directories( vcl::vcl INTERFACE ${vcl_SOURCE_DIR})
    # Per default the dg library includes "vcl/vectorclass.h" but the vcl folder
    # does not exist in the vectorclass github, so tell dg to directly include "vectorclass.h"
    target_compile_definitions( vcl::vcl INTERFACE VCL_NO_INCLUDE_PREFIX)
endif()

target_link_libraries(dg_dg INTERFACE vcl::vcl)

# Add cccl
CPMAddPackage( "gh:NVIDIA/cccl#v2.8.0")

target_link_libraries(dg_dg INTERFACE CCCL::CCCL)

if(CCCL_THRUST_DEVICE_SYSTEM STREQUAL "CUDA" OR CCCL_THRUST_DEVICE_SYSTEM STREQUAL "")
  # CUDA language must be enabled if device is CUDA
  target_link_libraries(dg_dg INTERFACE CUDA::cusparse)
  target_compile_features(dg_dg INTERFACE cuda_std_17)
endif()
# https://stackoverflow.com/questions/53256405/howto-pass-flag-to-nvcc-compiler-in-cmake
target_compile_options(dg_dg INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)


# MW: cccl already links OpenMP
# if(CCCL_THRUST_DEVICE_SYSTEM STREQUAL "OMP")
#   find_package(OpenMP REQUIRED)
#   target_link_libraries(dg_dg INTERFACE OpenMP::OpenMP_CXX)
# endif()

# =============================================
# Set up benchmark and test targets
# MW a custom target is always out of date and has no output file
if(FELTOR_BUILD_BENCHMARKS)
  add_custom_target(dg_benchmarks)
endif()
if(FELTOR_BUILD_TESTS)
  add_custom_target(dg_tests)
endif()

# ============================================
# Add file utilities
# Defines the dg_file target

add_subdirectory(file)

# ============================================
# Add geometries utilities
# Defines the dg_geometries target

add_subdirectory(geometries)

# ============================================
# Add matrix utilities
# Defines the dg_matrix target

option(FELTOR_DG_WITH_MATRIX "Configure dg_matrix target. Depends on Boost and Lapack." ON)
if(FELTOR_DG_WITH_MATRIX)
    add_subdirectory(matrix)
endif()


# ============================================
# Add dg benchmarks

if(FELTOR_BUILD_BENCHMARKS)
    include(FeltorBenchmarks)
    # There are no backend and topology benchmarks
    # All benchmarks can be compiled for MPI
    foreach(benchmark_path IN LISTS DG_BENCHMARKS)
        add_dg_benchmark("${benchmark_path}" "" ${FELTOR_WITH_MPI} target_name )
    endforeach()
endif()

# ============================================
# Add dg tests
# We need to parse the paths to be sure to put topology and backend tests in the right folder

if(FELTOR_BUILD_TESTS)
    include(FeltorTests)
    if(FELTOR_WITH_MPI)
        foreach(test_path IN LISTS DG_MPI_TESTS)
            cmake_path( REMOVE_FILENAME test_path OUTPUT_VARIABLE outvar)
            if( outvar STREQUAL "topology/")
                add_dg_test("${test_path}" "topology" ON target_name )
            elseif( outvar STREQUAL "backend/")
                add_dg_test("${test_path}" "backend" ON target_name )
            else()
                add_dg_test("${test_path}" "" ON target_name )
            endif()
        endforeach()
    else()
        foreach(test_path IN LISTS DG_TESTS)
            cmake_path( REMOVE_FILENAME test_path OUTPUT_VARIABLE outvar)
            if( outvar STREQUAL "topology/")
                add_dg_test("${test_path}" "topology" OFF target_name )
            elseif( outvar STREQUAL "backend/")
                add_dg_test("${test_path}" "backend" OFF target_name )
            else()
                add_dg_test("${test_path}" "" OFF target_name )
            endif()
        endforeach()
    endif()
endif()

