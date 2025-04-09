# inc/dg/CMakeLists.txt

set(DG_HEADERS
  adaptive.h
  advection.h
  algorithm.h
  andersonacc.h
  arakawa.h
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
  backend/exblas/ExSUM.FPE.hpp
  backend/exblas/accumulate.h
  backend/exblas/config.h
  backend/exblas/exblas.h
  backend/exblas/exblas_doc.h
  backend/exblas/exdot_omp.h
  backend/exblas/exdot_serial.h
  backend/exblas/fpedot_omp.h
  backend/exblas/fpedot_serial.h
  backend/exblas/mpi_accumulate.h
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
  backend/sparseblockmat.h
  backend/sparseblockmat_cpu_kernels.h
  backend/sparseblockmat_omp_kernels.h
  backend/sparsematrix.h
  backend/sparsematrix_cpu.h
  backend/sparsematrix_omp.h
  backend/tensor_traits.h
  backend/tensor_traits_cusp.h
  backend/tensor_traits_scalar.h
  backend/tensor_traits_std.h
  backend/tensor_traits_thrust.h
  backend/timer.h
  backend/typedefs.h
  backend/vector_categories.h
  backend/view.h
  bicgstabl.h
  blas.h
  blas1.h
  blas2.h
  chebyshev.h
  dg_doc.h
  elliptic.h
  enums.h
  eve.h
  exblas/exblas.h
  extrapolation.h
  functors.h
  geometries/geometries.h
  helmholtz.h
  implicit.h
  lgmres.h
  matrix/matrix.h
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
  topology/derivatives.h
  topology/derivativesA.h
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

if(FELTOR_USE_GPU)
  list(APPEND DG_HEADERS
    backend/blas1_cuda.cuh
    backend/exblas/accumulate.cuh
    backend/exblas/exdot_cuda.cuh
    backend/exblas/fpedot_cuda.cuh
    backend/exblas/mylibm.cuh
    backend/exblas/mylibm.cuh
    backend/sparseblockmat_gpu_kernels.cuh
    backend/sparseblockmat_gpu_kernels.cuh
    backend/sparsematrix_gpu.cuh
  )
endif()

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
  topology/derivativesX_t.cpp
  topology/derivatives_t.cpp
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

# ============================================
# Create DG library target

# Create the dg target library.
# As a header-only library, this must be an interface library.
# This means it has no sources, but can link/be linked to other
# libraries, have dependencies, etc. These properties are then
# inherited by any target that links to it.
add_library(dg INTERFACE ${DG_HEADERS})
# Ensure that `#include "dg/..."` works
# TODO remove this after other libs have their own CMakeLists.txt
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
add_library(dg::dg ALIAS dg)

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
  target_link_libraries(dg INTERFACE MPI::MPI_CXX)
endif()

if(FELTOR_USE_OMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(dg INTERFACE OpenMP::OpenMP_CXX)
endif()

# =============================================
# Set up benchmark and test targets
if(FELTOR_BUILD_BENCHMARKS)
  add_custom_target(dg_benchmarks)
endif()
if(FELTOR_BUILD_TESTS)
  add_custom_target(dg_tests)
endif()

# ============================================
# Add file utilities
# Defines the dg::file target

add_subdirectory(file)

# ============================================
# Add dg benchmarks

if(FELTOR_BUILD_BENCHMARKS)
  include(FeltorBenchmarks)
  foreach(benchmark_path IN LISTS DG_BENCHMARKS)
    add_dg_benchmark("${benchmark_path}")
  endforeach()
endif()

# ============================================
# Add dg tests

if(FELTOR_BUILD_TESTS)
  include(FeltorTests)
  foreach(test_path IN LISTS DG_TESTS)
    add_dg_test("${test_path}")
  endforeach()
  if(FELTOR_USE_MPI)
    foreach(test_path IN LISTS DG_MPI_TESTS)
      add_dg_mpi_test("${test_path}")
    endforeach()
  endif()
endif()

