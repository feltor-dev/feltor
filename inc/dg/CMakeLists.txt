# Sets the following variables
# - DG_HEADERS
# - DG_BENCHMARKS
# - DG_TESTS
# - DG_MPI_TESTS

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
  file/file.h
  file/json_utilities.h
  file/nc_utilities.h
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

# Modify each to use absolute path

function(absolutise PATH_LIST)
  # Pass in the variable name. Overwrites the variable.
  set(ABSOLUTISED "")
  foreach(REL_PATH IN LISTS ${PATH_LIST})
    cmake_path(ABSOLUTE_PATH REL_PATH NORMALIZE OUTPUT_VARIABLE ABS_PATH)
    list(APPEND ABSOLUTISED "${ABS_PATH}")
  endforeach()
  set(${PATH_LIST} "${ABSOLUTISED}" PARENT_SCOPE)
endfunction()

absolutise(DG_HEADERS)
absolutise(DG_BENCHMARKS)
absolutise(DG_TESTS)
absolutise(DG_MPI_TESTS)

# Expose the variables to the parent scope
set(DG_HEADERS "${DG_HEADERS}" PARENT_SCOPE)
set(DG_BENCHMARKS "${DG_BENCHMARKS}" PARENT_SCOPE)
set(DG_TESTS "${DG_TESTS}" PARENT_SCOPE)
set(DG_MPI_TESTS "${DG_MPI_TESTS}" PARENT_SCOPE)

# Check if paths are valid
function(check_paths PATH_LIST)
  foreach(PATH IN LISTS ${PATH_LIST})
    if(NOT EXISTS "${PATH}")
      message(WARNING "Header file ${PATH} does not exist.")
    endif()
  endforeach()
endfunction()

check_paths(DG_HEADERS)
check_paths(DG_BENCHMARKS)
check_paths(DG_TESTS)
check_paths(DG_MPI_TESTS)
