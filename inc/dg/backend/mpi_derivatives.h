#pragma once

#include "functions.h"
#include "sparseblockmat.h"
#include "derivatives.h"
#include "mpi_vector.h"
#include "mpi_matrix.h"

namespace dg{

namespace create{ 

namespace detail{ 


/**
* @brief Iterate through elements of a matrix
*
* stores all elements with column -1 or num_rows
* @param m The input matrix
*
* @return a newly created Coordinate matrix
*/
COO_SparseBlockMat save_outer_values(const SparseBlockMat& m)
{
    COO_SparseBlockMat mat();
    thrust::host_vector<double> data_element(m.n*m.n);
    for( unsigned i=0; i<m.num_rows; i++)
        for( unsigned d=0; d<m.blocks_per_line; d++)
        {
            if( m.cols_idx[i*m.blocks_per_line+d]==-1)
            {
                for( unsigned j=0; j<m.n*m.n; j++)
                    data_element[j] = m.data[ m.data_idx[i*m.blocks_per_line+d]*m.n*m.n + j];
                mat.add_value( i, 0, data_element);
            }
            if( m.cols_idx[i*m.blocks_per_line+d]==num_cols)
            {
                for( unsigned j=0; j<m.n*m.n; j++)
                    data_element[j] = m.data[ m.data_idx[i*m.blocks_per_line+d]*m.n*m.n + j];
                mat.add_value( i, 1, data_element);
            }
        }
    return mat;
}
void remove_outer_values(SparseBlockMat& m)
{
    int index = m.data.size()/ m.n/m.n;
    thrust::host_vector<double> zero(m.n*m.n, 0);
    //1. add a zero element to the data array
    m.data.insert( m.data.end(), zero.begin(), zero.end()); 
    //2. search and destroy columns that are not inside 
    for( unsigned i=0; i<m.num_rows; i++)
        for( unsigned d=0; d<m.blocks_per_line; d++)
        {
            if( m.cols_idx[i*m.blocks_per_line+d]==-1)
            {
                m.data_idx[i*m.blocks_per_line+d] = index; //
                m.cols_idx[i*m.blocks_per_line+d] = 0;
            }
            if( m.cols_idx[i*m.blocks_per_line+d]==m.num_cols)
            {
                m.data_idx[i*m.blocks_per_line+d] = index;
                m.cols_idx[i*m.blocks_per_line+d] = m.num_cols-1;
            }
        }
}
//distribute the inner block to howmany processes
/**
* @brief Reduce a global matrix into equal chunks among mpi processes
*
* copies all data elements. 
* grabs the right chunk of column and data indices and remaps the column indices to vector with ghostcells
* @param coord The mpi proces coordinate of the proper dimension
* @param howmany[3] # of processes 0 is left, 1 is the middle, 2 is right
*/
SparseBlockMat distribute_rows( const SparseBlockMat& src, int coord, const int* howmany)
{
    if( howmany[1] == 1)
    {
        SparseBlockMat temp(src);
        temp.left = temp.left/howmany[0];
        temp.right = temp.right/howmany[2];
        return temp;
    }
    assert( num_rows == num_cols);
    int chunk_size = num_rows/howmany[1];
    SparseBlockMat temp(src.chunk_size, src.chunk_size, src.blocks_per_line, src.data.size()/(src.n*src.n), src.n);
    temp.left = src.left/howmany[0];
    temp.right = src.right/howmany[2];
    //first copy data elements (even though not all might be needed it doesn't slow down things either)
    for( unsigned  i=0; i<src.data.size(); i++)
        temp.data[i] = src.data[i];
    //now grab the right chunk of cols and data indices
    for( unsigned i=0; i<temp.cols_idx.size(); i++)
    {
        temp.data_idx[i] = src.data_idx[ coord*(chunk_size*src.blocks_per_line)+i];
        temp.cols_idx[i] = src.cols_idx[ coord*(chunk_size*src.blocks_per_line)+i];
        //data indices are correct but cols are still the global indices (remapping a bit clumsy)
        //first in the zeroth line the col idx might be (global)num_cols - 1 -> map that to -1
        if( i/src.blocks_per_line == 0 && temp.cols_idx[i] == src.num_cols-1) temp.cols_idx[i] = -1; 
        //second in the last line the col idx mighty be 0 -> map to (global)num_cols
        if( (int)i/src.blocks_per_line == temp.num_rows-1 && temp.cols_idx[i] == 0) temp.cols_idx[i] = src.num_cols;  
        //Elements are now in the range -1, 0, 1,..., (global)num_cols
        //now shift this range to chunk range -1,..,chunk_size
        temp.cols_idx[i] = (temp.cols_idx[i] - coord*chunk_size ); 
    }
    return temp;
}

} //namespace detail

RowColDistMat< SparseBlockMatELL, SparseBlockMatCOO, NNCH> dx( const MPI_Grid2d& g, bc bcx, direction dir = centered)
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
    SparseBlockMat inner = detail::distribute_rows(dx, coords[0], howmany);
    COO_SparseBlockMat outer = detail::save_outer_values(inner);
    detail::remove_outer_values( inner);

    int vector_dimensions[] = {(int)(g.n()*g.Nx()), (int)(g.n()*g.Ny()), 1}; //x, y, z
    NNCH c( g.n(), vector_dimensions, comm, 0);
    if( howmany[1] == 1)
    {
        COO_SparseBlockMat outer();
        return RowColDistMat<>( dx, outer, c);
        
    }
    RowColDistMat<SparseBlockMat, NNCH> matrix( inner, outer, c);
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
    int vector_dimensions[] = {(int)(g.n()*g.Nx()), (int)(g.n()*g.Ny()), 1}; //x, y, z
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
    int vector_dimensions[] = {(int)(g.n()*g.Nx()), (int)(g.n()*g.Ny()), 1}; //x, y, z
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
    int vector_dimensions[] = {(int)(g.n()*g.Nx()), (int)(g.n()*g.Ny()), 1}; //x, y, z
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
    int vector_dimensions[] = {(int)(g.n()*g.Nx()), (int)(g.n()*g.Ny()), (int)(g.Nz())}; //x, y, z
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
    int vector_dimensions[] = {(int)(g.n()*g.Nx()), (int)(g.n()*g.Ny()), (int)(g.Nz())}; //x, y, z
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
    int vector_dimensions[] = {(int)(g.n()*g.Nx()), (int)(g.n()*g.Ny()), (int)(g.Nz())}; //x, y, z
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
    int vector_dimensions[] = {(int)(g.n()*g.Nx()), (int)(g.n()*g.Ny()), (int)(g.Nz())}; //x, y, z
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
    int vector_dimensions[] = {(int)(g.n()*g.Nx()), (int)(g.n()*g.Ny()), (int)(g.Nz())}; //x, y, z
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
    int vector_dimensions[] = {(int)(g.n()*g.Nx()), (int)(g.n()*g.Ny()), (int)(g.Nz())}; //x, y, z
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
