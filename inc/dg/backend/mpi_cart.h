#pragma once
#include <algorithm>


namespace dg
{

// Wrap all MPI_Cart functions such that Cartesian topology becomes column
// major ordering instead of row major
//
// Motivation: usually MPI process ranks fill HPC compute nodes in sequence,
// e.g. the first four ranks are on node 0, the next four on node 1 and so on.
// At the same time we want that in our 3d tokamak simulations all processes
// that partition a 2d plane lie on the same node (to reduce communication
// times for perp derivatives). However, \c MPI_Cart_create assigns ranks to
// dimensions in row-major ordering such that the last direction is the fastest
// varying one, e.g. in 1 2 4 the first four ranks all have different z-values
// Pinning processes differently is possible but usually a hassle (and we
// should make our GPU assignment more sophisticated in mpi_init.h). This
// solution simply makes the ordering column major by reversing the order to
// dimensions for all MPI_Cart_* calls.
//
// Rationale: The following functions capture the transposition and handle it
// correctly for the user. If they are always used instead of the originals one
// does not need to think about the transposition and simply ignore it.
///@cond

inline int mpi_cart_coords( MPI_Comm comm, int rank, int maxdims, int coords[])
{
    int re = MPI_Cart_coords( comm, rank, maxdims, coords);
    std::reverse( coords, coords + maxdims);
    return re;
}

inline int mpi_cart_create( MPI_Comm comm_old, int ndims, int dims[],
    const int periods[], int reorder, MPI_Comm * comm_cart
)
{
    std::vector<int> rev_dims(ndims), rev_periods(ndims);
    std::reverse_copy( dims, dims + ndims, rev_dims.begin());
    std::reverse_copy( periods, periods + ndims, rev_periods.begin());
    return MPI_Cart_create( comm_old, ndims, &rev_dims[0], &rev_periods[0], reorder, comm_cart);
}

inline int mpi_cart_get( MPI_Comm comm, int maxdims, int dims[], int periods[], int coords[])
{
    int re = MPI_Cart_get( comm, maxdims, dims, periods, coords);
    std::reverse( dims, dims + maxdims);
    std::reverse( periods, periods + maxdims);
    std::reverse( coords, coords + maxdims);
    return re;
}

inline int mpi_cart_rank( MPI_Comm comm, int coords[], int *rank)
{
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    std::vector<int> rev_coords(ndims);
    std::reverse_copy( coords, coords + ndims, rev_coords.begin());
    return MPI_Cart_rank( comm, &rev_coords[0], rank);
}

inline int mpi_cart_shift( MPI_Comm comm, int direction, int disp, int *rank_source, int *rank_dest)
{
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    direction = ndims - 1 - direction;
    return MPI_Cart_shift( comm, direction, disp, rank_source, rank_dest);
}

inline int mpi_cart_sub( MPI_Comm comm, const int remain_dims[], MPI_Comm *comm_new)
{
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    std::vector<int> rev_remains(ndims);
    std::reverse_copy( remain_dims, remain_dims + ndims, rev_remains.begin());
    return MPI_Cart_sub( comm, &rev_remains[0], comm_new);
}

inline int mpi_cartdim_get( MPI_Comm comm, int *ndims) { return MPI_Cartdim_get( comm, ndims);}
///@endcond

} // namespace dg
