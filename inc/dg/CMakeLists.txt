# Sets the following variables
# - DG_HEADERS
# - DG_BENCHMARKS
# - DG_MPI_BENCHMARKS
# - DG_TESTS
# - DG_DRAW_TESTS
# - DG_MPI_TESTS

set(DG_CXX_HEADERS
  adaptive.h
  advection.h
  algorithm.h
  andersonacc.h
  arakawa.h
  backend/average_cpu.h
  backend/average_dispatch.h
  backend/average_omp.h
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
  backend/exblas/accumulate.h
  backend/exblas/config.h
  backend/exblas/exblas_doc.h
  backend/exblas/exblas.h
  backend/exblas/exdot_omp.h
  backend/exblas/exdot_serial.h
  backend/exblas/ExSUM.FPE.hpp
  backend/exblas/mpi_accumulate.h
  backend/exblas/mylibm.hpp
  backend/exceptions.h
  backend/execution_policy.h
  backend/matrix_categories.h
  backend/memory.h
  backend/mpi_collective.h
  backend/mpi_communicator.h
  backend/mpi_init.h
  backend/mpi_matrix.h
  backend/mpi_vector.h
  backend/predicate.h
  backend/scalar_categories.h
  backend/sparseblockmat.h
  backend/sparseblockmat_omp_kernels.h
  backend/tensor_traits_cusp.h
  backend/tensor_traits.h
  backend/tensor_traits_scalar.h
  backend/tensor_traits_std.h
  backend/tensor_traits_thrust.h
  backend/timer.h
  backend/transpose.h
  backend/typedefs.h
  backend/vector_categories.h
  backend/view.h
  bicgstabl.h
  blas1.h
  blas2.h
  blas.h
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
  precond.h
  refined_elliptic.h
  runge_kutta.h
  simpsons.h
  subroutines.h
  tableau.h
  topology/average.h
  topology/average_mpi.h
  topology/base_geometry.h
  topology/base_geometryX.h
  topology/derivatives.h
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
  topology/mpi_base.h
  topology/mpi_derivatives.h
  topology/mpi_evaluation.h
  topology/mpi_grid.h
  topology/mpi_projection.h
  topology/mpi_weights.h
  topology/multiply.h
  topology/operator.h
  topology/operator_tensor.h
  topology/projection.h
  topology/projectionX.h
  topology/refined_grid.h
  topology/refined_gridX.h
  topology/split_and_join.h
  topology/stencil.h
  topology/tensor.h
  topology/topological_traits.h
  topology/transform.h
  topology/weights.h
  topology/weightsX.h
  topology/xspacelib.h
  # CUDA headers
  backend/average_gpu.cuh
  backend/blas1_cuda.cuh
  backend/exblas/accumulate.cuh
  backend/exblas/exdot_cuda.cuh
  backend/exblas/mylibm.cuh
  backend/sparseblockmat.cuh
  backend/sparseblockmat_gpu_kernels.cuh
  topology/ell_interpolation.cuh
)

set(DG_BENCHMARKS
  arakawa3d_b.cu
  arakawa_b.cu
  blas_b.cu
  cg2d_b.cu
  cg3d_b.cu
  elliptic2d_b.cu
  elliptic_b.cu
  ellipticX2d_b.cu
  helmholtz_b.cu
  helmholtzg2_b.cu
  multigrid_b.cu
  poisson_b.cu
  precond_b.cu
  refined_elliptic2d_b.cu
  topology/average_b.cu
  topology/derivatives_b.cu
  topology/ell_interpolation_b.cu
  topology/fem_b.cu
  topology/multiply_b.cu
)

set(DG_MPI_BENCHMARKS
  arakawa3d_mpib.cu
  arakawa_mpib.cu
  blas_mpib.cu
  cg2d_mpib.cu
  cg3d_mpib.cu
  cluster_mpib.cu
  elliptic2d_mpib.cu
  elliptic_mpib.cu
  topology/average_mpib.cu
  topology/derivatives_mpib.cu
)

set(DG_TESTS
  adaptive_t.cu
  andersonacc_t.cu
  arakawa_t.cu
  backend/memory_t.cu
  backend/view_t.cu
  blas1_t.cu
  blas_t.cu
  cg2d_t.cu
  cg3d_t.cu
  extrapolation_t.cu
  helmholtzg2_t.cu
  helmholtz_t.cu
  implicit_t.cu
  multistep_t.cu
  poisson_t.cu
  precond_t.cu
  runge_kutta_t.cu
  simpsons_t.cu
  topology/average_t.cu
  topology/derivatives_t.cu
  topology/derivativesX_t.cu
  topology/dlt_t.cu
  topology/dx_t.cu
  topology/ell_interpolation_t.cu
  topology/evaluation_t.cu
  topology/fem_t.cu
  topology/filter_t.cu
  topology/geometry_t.cu
  topology/grid_t.cu
  topology/interpolation_t.cu
  topology/multiply_t.cu
  topology/operator_t.cu
  topology/operator_tensor_t.cu
  topology/projection_t.cu
  topology/refined_grid_t.cu
  topology/stencil_t.cu
  topology/tensor_t.cu
  topology/transform_t.cu
)

set(DG_DRAW_TESTS
  bathRZ_t.cu
  topology/backscatter_t.cu
)

set(DG_MPI_TESTS
  arakawa_mpit.cu
  backend/mpi_collective_mpit.cu
  blas1_mpit.cu
  blas_mpit.cu
  cg2d_mpit.cu
  ping_mpit.cu
  topology/average_mpit.cu
  topology/derivatives_mpit.cu
  topology/evaluation_mpit.cu
  topology/filter_mpit.cu
  topology/geometry_mpit.cu
  topology/interpolation_mpit.cu
  topology/projection_mpit.cu
  topology/stencil_mpit.cu
)

# Modify each to use absolute path

function(absolutize PATH_LIST)
  # Pass in the variable name. Overwrites the variable.
  set(ABSOLUTIZED "")
  foreach(REL_PATH IN LISTS ${PATH_LIST})
    cmake_path(ABSOLUTE_PATH REL_PATH NORMALIZE OUTPUT_VARIABLE ABS_PATH)
    list(APPEND ABSOLUTIZED "${ABS_PATH}")
  endforeach()
  set(${PATH_LIST} "${ABSOLUTIZED}" PARENT_SCOPE)
endfunction()

absolutize(DG_HEADERS)
absolutize(DG_BENCHMARKS)
absolutize(DG_MPI_BENCHMARKS)
absolutize(DG_TESTS)
absolutize(DG_DRAW_TESTS)
absolutize(DG_MPI_TESTS)

# Expose the variables to the parent scope
set(DG_HEADERS "${DG_HEADERS}" PARENT_SCOPE)
set(DG_BENCHMARKS "${DG_BENCHMARKS}" PARENT_SCOPE)
set(DG_MPI_BENCHMARKS "${DG_MPI_BENCHMARKS}" PARENT_SCOPE)
set(DG_TESTS "${DG_TESTS}" PARENT_SCOPE)
set(DG_DRAW_TESTS "${DG_DRAW_TESTS}" PARENT_SCOPE)
set(DG_MPI_TESTS "${DG_MPI_TESTS}" PARENT_SCOPE)
