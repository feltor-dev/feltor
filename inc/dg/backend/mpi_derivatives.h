#pragma once

#include "functions.h"
#include "sparseblockmat.h"
#include "derivatives.h"
#include "mpi_vector.h"
#include "mpi_matrix.h"

namespace dg{

namespace create{ 


RowDistMat< SparseBlockMat, NNCH> dx( const MPI_Grid2d& g, bc bcx, direction dir = centered)
{
    SparseBlockMat dx = dg::create::dx( g.global(), bcx, dir);
    //get cartesian structure of mpi grid
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 2);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);
    int howmany[] = {dims[1], dims[0], 1}; //left, middle, right
    dx.distribute_rows( coords[0], howmany);
    int vector_dimensions[] = {g.n()*g.Nx(), g.n()*g.Ny(), 1}; //x, y, z
    NNCH c( g.n(), vector_dimensions, comm, 0);
    RowDistMat<SparseBlockMat, NNCH> matrix( dx, c);
    return matrix;
}
RowDistMat< SparseBlockMat, NNCH> dy( const MPI_Grid2d& g, bc bcy, direction dir = centered)
{
    SparseBlockMat dy = dg::create::dy( g.global(), bcy, dir);
    //get cartesian structure of mpi grid
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 2);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);
    int howmany[] = {1, dims[1], dims[0]};
    int vector_dimensions[] = {g.n()*g.Nx(), g.n()*g.Ny(), 1}; //x, y, z
    dy.distribute_rows( coords[1], howmany);
    NNCH c( g.n(), vector_dimensions, comm, 1);
    RowDistMat<SparseBlockMat, NNCH> matrix( dy, c);
    return matrix;
}
RowDistMat< SparseBlockMat, NNCH> jumpX( const MPI_Grid2d& g, bc bcx, direction dir = centered)
{
    SparseBlockMat jumpX = dg::create::jumpX( g.global(), bcx);
    //get cartesian structure of mpi grid
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 2);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);
    int howmany[] = {dims[1], dims[0], 1};
    int vector_dimensions[] = {g.n()*g.Nx(), g.n()*g.Ny(), 1}; //x, y, z
    jumpX.distribute_rows( coords[0], howmany);
    NNCH c( g.n(), vector_dimensions, comm, 0);
    RowDistMat<SparseBlockMat, NNCH> matrix( jumpX, c);
    return matrix;
}
RowDistMat< SparseBlockMat, NNCH> jumpY( const MPI_Grid2d& g, bc bcy, direction dir = centered)
{
    SparseBlockMat jumpY = dg::create::jumpY( g.global(), bcy);
    //get cartesian structure of mpi grid
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 2);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);
    int howmany[] = {1, dims[1], dims[0]};
    int vector_dimensions[] = {g.n()*g.Nx(), g.n()*g.Ny(), 1}; //x, y, z
    jumpY.distribute_rows( coords[1], howmany);
    NNCH c( g.n(), vector_dimensions, comm, 1);
    RowDistMat<SparseBlockMat, NNCH> matrix( jumpY, c);
    return matrix;
}
RowDistMat< SparseBlockMat, NNCH> dx( const MPI_Grid3d& g, bc bcx, direction dir = centered)
{
    SparseBlockMat dx = dg::create::dx( g.global(), bcx, dir);
    //get cartesian structure of mpi grid
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);
    int howmany[] = {dims[2]*dims[1], dims[0], 1};
    int vector_dimensions[] = {g.n()*g.Nx(), g.n()*g.Ny(), g.Nz()}; //x, y, z
    dx.distribute_rows( coords[0], howmany);
    NNCH c( g.n(), vector_dimensions, comm, 0);
    RowDistMat<SparseBlockMat, NNCH> matrix( dx, c);
    return matrix;
}
RowDistMat< SparseBlockMat, NNCH> dy( const MPI_Grid3d& g, bc bcy, direction dir = centered)
{
    SparseBlockMat dy = dg::create::dy( g.global(), bcy, dir);
    //get cartesian structure of mpi grid
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);
    int howmany[] = {dims[2], dims[1], dims[0]};
    dy.distribute_rows( coords[1], howmany);
    int vector_dimensions[] = {g.n()*g.Nx(), g.n()*g.Ny(), g.Nz()}; //x, y, z
    NNCH c( g.n(), vector_dimensions, comm, 1);
    RowDistMat<SparseBlockMat, NNCH> matrix( dy, c);
    return matrix;
}
RowDistMat< SparseBlockMat, NNCH> dz( const MPI_Grid3d& g, bc bcz, direction dir = centered)
{
    SparseBlockMat dz = dg::create::dz( g.global(), bcz, dir);
    //get cartesian structure of mpi grid
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);
    int howmany[] = {1, dims[2], dims[1]*dims[0]};
    int vector_dimensions[] = {g.n()*g.Nx(), g.n()*g.Ny(), g.Nz()}; //x, y, z
    dz.distribute_rows( coords[2], howmany);
    NNCH c( 1, vector_dimensions, comm, 2);
    RowDistMat<SparseBlockMat, NNCH> matrix( dz, c);
    return matrix;
}
RowDistMat< SparseBlockMat, NNCH> jumpX( const MPI_Grid3d& g, bc bcx)
{
    SparseBlockMat jumpX = dg::create::jumpX( g.global(), bcx);
    //get cartesian structure of mpi grid
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);
    int howmany[] = {dims[2]*dims[1], dims[0], 1};
    int vector_dimensions[] = {g.n()*g.Nx(), g.n()*g.Ny(), g.Nz()}; //x, y, z
    jumpX.distribute_rows( coords[0], howmany);
    NNCH c( g.n(), vector_dimensions, comm, 0);
    RowDistMat<SparseBlockMat, NNCH> matrix( jumpX, c);
    return matrix;
}
RowDistMat< SparseBlockMat, NNCH> jumpY( const MPI_Grid3d& g, bc bcy)
{
    SparseBlockMat jumpY = dg::create::jumpY( g.global(), bcy);
    //get cartesian structure of mpi grid
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);
    int howmany[] = {dims[2], dims[1], dims[0]};
    int vector_dimensions[] = {g.n()*g.Nx(), g.n()*g.Ny(), g.Nz()}; //x, y, z
    jumpY.distribute_rows( coords[1], howmany);
    NNCH c( g.n(), vector_dimensions, comm, 1);
    RowDistMat<SparseBlockMat, NNCH> matrix( jumpY, c);
    return matrix;
}
RowDistMat< SparseBlockMat, NNCH> jumpZ( const MPI_Grid3d& g, bc bcz)
{
    SparseBlockMat jumpZ = dg::create::jumpZ( g.global(), bcz);
    //get cartesian structure of mpi grid
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);
    int howmany[] = {1, dims[2], dims[1]*dims[0]};
    int vector_dimensions[] = {g.n()*g.Nx(), g.n()*g.Ny(), g.Nz()}; //x, y, z
    jumpZ.distribute_rows( coords[2], howmany);
    NNCH c( 1, vector_dimensions, comm, 2);
    RowDistMat<SparseBlockMat, NNCH> matrix( jumpZ, c);
    return matrix;
}

/**
 * @brief Create 2d derivative in x-direction
 *
 * @param g The grid on which to create dx (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A mpi matrix in coordinate format
 */
RowDistMat<SparseBlockMat, NNCH> dx( const MPI_Grid2d& g, direction dir = centered)
{
    return dx( g, g.bcx(), dir);
}

/**
 * @brief Create 3d derivative in y-direction
 *
 * @param g The grid on which to create dy (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A mpi matrix in coordinate format
 */
RowDistMat<SparseBlockMat, NNCH> dx( const MPI_Grid3d& g, direction dir = centered)
{
    return dx( g, g.bcx(), dir);
}
RowDistMat<SparseBlockMat, NNCH> jumpX( const MPI_Grid2d& g)
{
    return jumpX( g, g.bcx());
}

RowDistMat<SparseBlockMat, NNCH> jumpX( const MPI_Grid3d& g)
{
    return jumpX( g, g.bcx());
}
/**
 * @brief Create 2d derivative in y-direction
 *
 * @param g The grid on which to create dx (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A mpi matrix in coordinate format
 */
RowDistMat<SparseBlockMat, NNCH> dy( const MPI_Grid2d& g, direction dir = centered)
{
    return dy( g, g.bcy(), dir);
}

RowDistMat<SparseBlockMat, NNCH> dy( const MPI_Grid3d& g, direction dir = centered)
{
    return dy( g, g.bcy(), dir);
}
RowDistMat<SparseBlockMat, NNCH> dz( const MPI_Grid3d& g, direction dir = centered)
{
    return dz( g, g.bcz(), dir);
}
RowDistMat<SparseBlockMat, NNCH> jumpY( const MPI_Grid2d& g)
{
    return jumpY( g, g.bcy());
}

RowDistMat<SparseBlockMat, NNCH> jumpY( const MPI_Grid3d& g)
{
    return jumpY( g, g.bcy());
}
RowDistMat<SparseBlockMat, NNCH> jumpZ( const MPI_Grid3d& g)
{
    return jumpZ( g, g.bcz());
}




} //namespace create
} //namespace dg
