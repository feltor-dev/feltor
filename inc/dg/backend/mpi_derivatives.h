#pragma once

#include "functions.h"
#include "sparseblockmat.h"
#include "mpi_vector.h"
#include "mpi_matrix.h"

namespace dg{
namespace create
{


RowDistMat< SparseBlockMat, NearestNeighborComm> dx( MPI_Grid2d& g, bc bcx, direction dir = centered)
{
    SparseBlockMat dx = dx( g.global(), bcx, direction);
    //get cartesian structure of mpi grid
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 2);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);
    int howmany[] = {dims[1], dims[0], 1};
    int vector_dimensions[] = {1, g.n()*g.Ny(), g.n()*g.Nx()};
    dx.distribute_rows( coords[0], dims);
    NearestNeighborComm<thrust::host_vector<double>, thrust::host_vector<int> > c( g.n(), vector_dimensions, comm, 0);
    RowDistMat<SparseBlockMat, NearestNeighborComm> matrix( dx, c);
    return matrix;
}
RowDistMat< SparseBlockMat, NearestNeighborComm> dy( MPI_Grid2d& g, bc bcy, direction dir = centered)
{
    SparseBlockMat dy = dy( g.global(), bcy, direction);
    //get cartesian structure of mpi grid
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 2);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);
    int howmany[] = {1, dims[1], dims[0]};
    int vector_dimensions[] = {1, g.n()*g.Ny(), g.n()*g.Nx()};
    dy.distribute_rows( coords[1], dims);
    NearestNeighborComm<thrust::host_vector<double>, thrust::host_vector<int> > c( g.n(), vector_dimensions, comm, 1);
    RowDistMat<SparseBlockMat, NearestNeighborComm> matrix( dx, c);
    return matrix;
}
RowDistMat< SparseBlockMat, NearestNeighborComm> jumpX( MPI_Grid2d& g, bc bcx, direction dir = centered)
{
    SparseBlockMat jumpX = jumpX( g.global(), bcx, direction);
    //get cartesian structure of mpi grid
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 2);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);
    int howmany[] = {dims[1], dims[0], 1};
    int vector_dimensions[] = {1, g.n()*g.Ny(), g.n()*g.Nx()};
    jumpX.distribute_rows( coords[0], dims);
    NearestNeighborComm<thrust::host_vector<double>, thrust::host_vector<int> > c( g.n(), vector_dimensions, comm, 0);
    RowDistMat<SparseBlockMat, NearestNeighborComm> matrix( jumpX, c);
    return matrix;
}
RowDistMat< SparseBlockMat, NearestNeighborComm> jumpY( MPI_Grid2d& g, bc bcy, direction dir = centered)
{
    SparseBlockMat jumpY = jumpY( g.global(), bcy, direction);
    //get cartesian structure of mpi grid
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 2);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);
    int howmany[] = {1, dims[1], dims[0]};
    int vector_dimensions[] = {1, g.n()*g.Ny(), g.n()*g.Nx()};
    jumpY.distribute_rows( coords[1], dims);
    NearestNeighborComm<thrust::host_vector<double>, thrust::host_vector<int> > c( g.n(), vector_dimensions, comm, 1);
    RowDistMat<SparseBlockMat, NearestNeighborComm> matrix( dx, c);
    return matrix;
}
RowDistMat< SparseBlockMat, NearestNeighborComm> dx( MPI_Grid3d& g, bc bcx, direction dir = centered)
{
    SparseBlockMat dx = dx( g.global(), bcx, direction);
    //get cartesian structure of mpi grid
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);
    int howmany[] = {dims[2]*dims[1], dims[0], 1};
    int vector_dimensions[] = {g.Nz(), g.n()*g.Ny(), g.n()*g.Nx()};
    dx.distribute_rows( coords[0], dims);
    NearestNeighborComm<thrust::host_vector<double>, thrust::host_vector<int> > c( g.n(), vector_dimensions, comm, 0);
    RowDistMat<SparseBlockMat, NearestNeighborComm> matrix( dx, c);
    return matrix;
}
RowDistMat< SparseBlockMat, NearestNeighborComm> dy( MPI_Grid3d& g, bc bcy, direction dir = centered)
{
    SparseBlockMat dy = dy( g.global(), bcy, direction);
    //get cartesian structure of mpi grid
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);
    int howmany[] = {dims[2], dims[1], dims[0]};
    int vector_dimensions[] = {g.Nz(), g.n()*g.Ny(), g.n()*g.Nx()};
    dy.distribute_rows( coords[1], dims);
    NearestNeighborComm<thrust::host_vector<double>, thrust::host_vector<int> > c( g.n(), vector_dimensions, comm, 1);
    RowDistMat<SparseBlockMat, NearestNeighborComm> matrix( dy, c);
    return matrix;
}
RowDistMat< SparseBlockMat, NearestNeighborComm> dz( MPI_Grid3d& g, bc bcz, direction dir = centered)
{
    SparseBlockMat dz = dz( g.global(), bcz, direction);
    //get cartesian structure of mpi grid
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);
    int howmany[] = {1, dims[2], dims[1]*dims[0]};
    int vector_dimensions[] = {g.Nz(), g.n()*g.Ny(), g.n()*g.Nx()};
    dz.distribute_rows( coords[2], dims);
    NearestNeighborComm<thrust::host_vector<double>, thrust::host_vector<int> > c( 1, vector_dimensions, comm, 2);
    RowDistMat<SparseBlockMat, NearestNeighborComm> matrix( dz, c);
    return matrix;
}
RowDistMat< SparseBlockMat, NearestNeighborComm> jumpX( MPI_Grid3d& g, bc bcx, direction dir = centered)
{
    SparseBlockMat jumpX = jumpX( g.global(), bcx, direction);
    //get cartesian structure of mpi grid
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);
    int howmany[] = {dims[2]*dims[1], dims[0], 1};
    int vector_dimensions[] = {g.Nz(), g.n()*g.Ny(), g.n()*g.Nx()};
    jumpX.distribute_rows( coords[0], dims);
    NearestNeighborComm<thrust::host_vector<double>, thrust::host_vector<int> > c( g.n(), vector_dimensions, comm, 0);
    RowDistMat<SparseBlockMat, NearestNeighborComm> matrix( jumpX, c);
    return matrix;
}
RowDistMat< SparseBlockMat, NearestNeighborComm> jumpY( MPI_Grid3d& g, bc bcy, direction dir = centered)
{
    SparseBlockMat jumpY = jumpY( g.global(), bcy, direction);
    //get cartesian structure of mpi grid
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);
    int howmany[] = {dims[2], dims[1], dims[0]};
    int vector_dimensions[] = {g.Nz(), g.n()*g.Ny(), g.n()*g.Nx()};
    jumpY.distribute_rows( coords[1], dims);
    NearestNeighborComm<thrust::host_vector<double>, thrust::host_vector<int> > c( g.n(), vector_dimensions, comm, 1);
    RowDistMat<SparseBlockMat, NearestNeighborComm> matrix( jumpY, c);
    return matrix;
}
RowDistMat< SparseBlockMat, NearestNeighborComm> jumpZ( MPI_Grid3d& g, bc bcz, direction dir = centered)
{
    SparseBlockMat jumpZ = jumpZ( g.global(), bcz, direction);
    //get cartesian structure of mpi grid
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);
    int howmany[] = {1, dims[2], dims[1]*dims[0]};
    int vector_dimensions[] = {g.Nz(), g.n()*g.Ny(), g.n()*g.Nx()};
    jumpZ.distribute_rows( coords[2], dims);
    NearestNeighborComm<thrust::host_vector<double>, thrust::host_vector<int> > c( 1, vector_dimensions, comm, 2);
    RowDistMat<SparseBlockMat, NearestNeighborComm> matrix( jumpZ, c);
    return matrix;
}


///@addtogroup highlevel
///@{


/**
 * @brief Create 2d derivative in x-direction
 *
 * @param g The grid on which to create dx (boundary condition is taken from here)
 * @param no use normed normally
             use not_normed if you know what you're doing
 * @param dir The direction of the first derivative
 *
 * @return A mpi matrix in coordinate format
 */
RowDistMat<SparseBlockMat, NearestNeighborComm> dx( const MPI_Grid2d& g, direction dir = centered)
{
    return dx( g, g.bcx(), dir);
}

/**
 * @brief Create 3d derivative in y-direction
 *
 * @param g The grid on which to create dy (boundary condition is taken from here)
 * @param no use normed normally
             use not_normed if you know what you're doing
 * @param dir The direction of the first derivative
 *
 * @return A mpi matrix in coordinate format
 */
RowDistMat<SparseBlockMat, NearestNeighborComm> dx( const MPI_Grid3d& g, direction dir = centered)
{
    return dx( g, g.bcx(), dir);
}
RowDistMat<SparseBlockMat, NearestNeighborComm> jumpX( const MPI_Grid2d& g, direction dir = centered)
{
    return jumpX( g, g.bcx(), dir);
}

RowDistMat<SparseBlockMat, NearestNeighborComm> jumpX( const MPI_Grid3d& g, direction dir = centered)
{
    return jumpX( g, g.bcx(), dir);
}
/**
 * @brief Create 2d derivative in y-direction
 *
 * @param g The grid on which to create dx (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A mpi matrix in coordinate format
 */
RowDistMat<SparseBlockMat, NearestNeighborComm> dy( const MPI_Grid2d& g, direction dir = centered)
{
    return dy( g, g.bcy(), dir);
}

RowDistMat<SparseBlockMat, NearestNeighborComm> dy( const MPI_Grid3d& g, direction dir = centered)
{
    return dy( g, g.bcy(), dir);
}
RowDistMat<SparseBlockMat, NearestNeighborComm> dz( const MPI_Grid3d& g, direction dir = centered)
{
    return dz( g, g.bcz(), dir);
}
RowDistMat<SparseBlockMat, NearestNeighborComm> jumpY( const MPI_Grid2d& g, direction dir = centered)
{
    return jumpY( g, g.bcy(), dir);
}

RowDistMat<SparseBlockMat, NearestNeighborComm> jumpY( const MPI_Grid3d& g, direction dir = centered)
{
    return jumpY( g, g.bcy(), dir);
}
RowDistMat<SparseBlockMat, NearestNeighborComm> jumpZ( const MPI_Grid3d& g, direction dir = centered)
{
    return jumpZ( g, g.bcz(), dir);
}



///@}

} //namespace create
} //namespace dg
