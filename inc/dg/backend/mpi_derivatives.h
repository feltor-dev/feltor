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
CooSparseBlockMat save_outer_values(const EllSparseBlockMat& m)
{
    //search outer values in m
    CooSparseBlockMat mat( m.num_rows, 2, m.n, m.left, m.right);
    int index = m.data.size()/ m.n/m.n;
    thrust::host_vector<double> data_element(m.n*m.n, 0), zero(data_element);
    m.data.insert( m.data.end(), zero.begin(), zero.end()); 
    for( unsigned i=0; i<m.num_rows; i++)
        for( unsigned d=0; d<m.blocks_per_line; d++)
        {
            if( m.cols_idx[i*m.blocks_per_line+d]==-1)
            {
                for( unsigned j=0; j<m.n*m.n; j++)
                    data_element[j] = m.data[ m.data_idx[i*m.blocks_per_line+d]*m.n*m.n + j];
                mat.add_value( i, 0, data_element);
                m.data_idx[i*m.blocks_per_line+d] = index; //
                m.cols_idx[i*m.blocks_per_line+d] = 0;
            }
            if( m.cols_idx[i*m.blocks_per_line+d]==num_cols)
            {
                for( unsigned j=0; j<m.n*m.n; j++)
                    data_element[j] = m.data[ m.data_idx[i*m.blocks_per_line+d]*m.n*m.n + j];
                mat.add_value( i, 1, data_element);
                m.data_idx[i*m.blocks_per_line+d] = index;
                m.cols_idx[i*m.blocks_per_line+d] = m.num_cols-1;
            }
        }
    return mat;
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
EllSparseBlockMat distribute_rows( const EllSparseBlockMat& src, int coord, const int* howmany)
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
    EllSparseBlockMat temp(src.chunk_size, src.chunk_size, src.blocks_per_line, src.data.size()/(src.n*src.n), src.n);
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

RowColDistMat< EllSparseBlockMat, CooSparseBlockMat, NNCH> dx( const MPI_Grid2d& g, bc bcx, direction dir = centered)
{
    EllSparseBlockMat matrix = dg::create::dx( g.global(), bcx, dir);
    int vector_dimensions[] = {(int)(g.n()*g.Nx()), (int)(g.n()*g.Ny()), 1}; //x, y, z
    NNCH c( g.n(), vector_dimensions, comm, 0);
    if( c.size()==0 )
    {
        CooSparseBlockMat outer();
        return RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH>( matrix, outer, c);
    }

    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 2);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {dims[1], dims[0], 1}; //left, middle, right
    EllSparseBlockMat inner = detail::distribute_rows(matrix, coords[0], howmany);
    CooSparseBlockMat outer = detail::save_outer_values(inner);

    return RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH>( inner, outer, c);
}
RowColDistMat< EllSparseBlockMat, CooSparseBlockMat, NNCH> dy( const MPI_Grid2d& g, bc bcy, direction dir = centered)
{
    EllSparseBlockMat matrix = dg::create::dy( g.global(), bcy, dir);
    int vector_dimensions[] = {(int)(g.n()*g.Nx()), (int)(g.n()*g.Ny()), 1}; //x, y, z
    NNCH c( g.n(), vector_dimensions, comm, 1);
    if( c.size()==0 )
    {
        CooSparseBlockMat outer();
        return RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH>( matrix, outer, c);
    }

    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 2);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {1, dims[1], dims[0]};
    EllSparseBlockMat inner = detail::distribute_rows(matrix, coords[1], howmany);
    CooSparseBlockMat outer = detail::save_outer_values(inner);

    return RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH>( inner, outer, c);
}
RowColDistMat< EllSparseBlockMat, CooSparseBlockMat, NNCH> jumpX( const MPI_Grid2d& g, bc bcx, direction dir = centered)
{
    EllSparseBlockMat matrix = dg::create::jumpX( g.global(), bcx, dir);
    int vector_dimensions[] = {(int)(g.n()*g.Nx()), (int)(g.n()*g.Ny()), 1}; //x, y, z
    NNCH c( g.n(), vector_dimensions, comm, 0);
    if( c.size()==0 )
    {
        CooSparseBlockMat outer();
        return RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH>( matrix, outer, c);
    }

    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 2);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {dims[1], dims[0], 1}; //left, middle, right
    EllSparseBlockMat inner = detail::distribute_rows(matrix, coords[0], howmany);
    CooSparseBlockMat outer = detail::save_outer_values(inner);

    return RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH>( inner, outer, c);
}
RowColDistMat< EllSparseBlockMat, CooSparseBlockMat, NNCH> jumpY( const MPI_Grid2d& g, bc bcy, direction dir = centered)
{
    EllSparseBlockMat matrix = dg::create::jumpY( g.global(), bcy, dir);
    int vector_dimensions[] = {(int)(g.n()*g.Nx()), (int)(g.n()*g.Ny()), 1}; //x, y, z
    NNCH c( g.n(), vector_dimensions, comm, 1);
    if( c.size()==0 )
    {
        CooSparseBlockMat outer();
        return RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH>( matrix, outer, c);
    }

    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 2);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {1, dims[1], dims[0]};
    EllSparseBlockMat inner = detail::distribute_rows(matrix, coords[1], howmany);
    CooSparseBlockMat outer = detail::save_outer_values(inner);

    return RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH>( inner, outer, c);
}
RowColDistMat< EllSparseBlockMat, CooSparseBlockMat, NNCH> dx( const MPI_Grid3d& g, bc bcx, direction dir = centered)
{
    EllSparseBlockMat matrix = dg::create::dx( g.global(), bcx, dir);
    int vector_dimensions[] = {(int)(g.n()*g.Nx()), (int)(g.n()*g.Ny()), (int)(g.Nz())}; //x, y, z
    NNCH c( g.n(), vector_dimensions, comm, 0);
    if( c.size()==0 )
    {
        CooSparseBlockMat outer();
        return RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH>( matrix, outer, c);
    }

    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {dims[2]*dims[1], dims[0], 1};
    EllSparseBlockMat inner = detail::distribute_rows(matrix, coords[0], howmany);
    CooSparseBlockMat outer = detail::save_outer_values(inner);

    return RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH>( inner, outer, c);
}
RowColDistMat< EllSparseBlockMat, CooSparseBlockMat, NNCH> dy( const MPI_Grid3d& g, bc bcy, direction dir = centered)
{
    EllSparseBlockMat matrix = dg::create::dy( g.global(), bcy, dir);
    int vector_dimensions[] = {(int)(g.n()*g.Nx()), (int)(g.n()*g.Ny()), (int)(g.Nz())}; //x, y, z
    NNCH c( g.n(), vector_dimensions, comm, 1);
    if( c.size()==0 )
    {
        CooSparseBlockMat outer();
        return RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH>( matrix, outer, c);
    }

    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {dims[2], dims[1], dims[0]};
    EllSparseBlockMat inner = detail::distribute_rows(matrix, coords[1], howmany);
    CooSparseBlockMat outer = detail::save_outer_values(inner);

    return RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH>( inner, outer, c);
}
RowColDistMat< EllSparseBlockMat, CooSparseBlockMat, NNCH> dz( const MPI_Grid3d& g, bc bcz, direction dir = centered)
{
    EllSparseBlockMat matrix = dg::create::dz( g.global(), bcz, dir);
    int vector_dimensions[] = {(int)(g.n()*g.Nx()), (int)(g.n()*g.Ny()), (int)(g.Nz())}; //x, y, z
    NNCH c( 1, vector_dimensions, comm, 2);
    if( c.size()==0 )
    {
        CooSparseBlockMat outer();
        return RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH>( matrix, outer, c);
    }

    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {1, dims[2], dims[1]*dims[0]};
    EllSparseBlockMat inner = detail::distribute_rows(matrix, coords[2], howmany);
    CooSparseBlockMat outer = detail::save_outer_values(inner);

    return RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH>( inner, outer, c);
}
RowColDistMat< EllSparseBlockMat, CooSparseBlockMat, NNCH> jumpX( const MPI_Grid3d& g, bc bcx)
{
    EllSparseBlockMat matrix = dg::create::jumpX( g.global(), bcx);
    int vector_dimensions[] = {(int)(g.n()*g.Nx()), (int)(g.n()*g.Ny()), (int)(g.Nz())}; //x, y, z
    NNCH c( g.n(), vector_dimensions, comm, 0);
    if( c.size()==0 )
    {
        CooSparseBlockMat outer();
        return RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH>( matrix, outer, c);
    }

    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {dims[2]*dims[1], dims[0], 1};
    EllSparseBlockMat inner = detail::distribute_rows(matrix, coords[0], howmany);
    CooSparseBlockMat outer = detail::save_outer_values(inner);

    return RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH>( inner, outer, c);
}
RowColDistMat< EllSparseBlockMat, CooSparseBlockMat, NNCH> jumpY( const MPI_Grid3d& g, bc bcy)
{
    EllSparseBlockMat matrix = dg::create::jumpY( g.global(), bcy);
    int vector_dimensions[] = {(int)(g.n()*g.Nx()), (int)(g.n()*g.Ny()), (int)(g.Nz())}; //x, y, z
    NNCH c( g.n(), vector_dimensions, comm, 1);
    if( c.size()==0 )
    {
        CooSparseBlockMat outer();
        return RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH>( matrix, outer, c);
    }

    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {dims[2], dims[1], dims[0]};
    EllSparseBlockMat inner = detail::distribute_rows(matrix, coords[1], howmany);
    CooSparseBlockMat outer = detail::save_outer_values(inner);

    return RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH>( inner, outer, c);
}
RowColDistMat< EllSparseBlockMat, CooSparseBlockMat, NNCH> jumpZ( const MPI_Grid3d& g, bc bcz)
{
    EllSparseBlockMat matrix = dg::create::jumpZ( g.global(), bcz);
    int vector_dimensions[] = {(int)(g.n()*g.Nx()), (int)(g.n()*g.Ny()), (int)(g.Nz())}; //x, y, z
    NNCH c( 1, vector_dimensions, comm, 2);
    if( c.size()==0 )
    {
        CooSparseBlockMat outer();
        return RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH>( matrix, outer, c);
    }

    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {1, dims[2], dims[1]*dims[0]};
    EllSparseBlockMat inner = detail::distribute_rows(matrix, coords[2], howmany);
    CooSparseBlockMat outer = detail::save_outer_values(inner);

    return RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH>( inner, outer, c);
}

/**
 * @brief Create 2d derivative in x-direction
 *
 * @param g The grid on which to create dx (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A mpi matrix 
 */
RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH> dx( const MPI_Grid2d& g, direction dir = centered)
{
    return dx( g, g.bcx(), dir);
}

/**
 * @brief Create 3d derivative in x-direction
 *
 * @param g The grid on which to create dx (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A mpi matrix 
 */
RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH> dx( const MPI_Grid3d& g, direction dir = centered)
{
    return dx( g, g.bcx(), dir);
}
RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH> jumpX( const MPI_Grid2d& g)
{
    return jumpX( g, g.bcx());
}

RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH> jumpX( const MPI_Grid3d& g)
{
    return jumpX( g, g.bcx());
}
/**
 * @brief Create 2d derivative in y-direction
 *
 * @param g The grid on which to create dy (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A mpi matrix
 */
RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH> dy( const MPI_Grid2d& g, direction dir = centered)
{
    return dy( g, g.bcy(), dir);
}

/**
 * @brief Create 3d derivative in y-direction
 *
 * @param g The grid on which to create dy (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A mpi matrix 
 */
RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH> dy( const MPI_Grid3d& g, direction dir = centered)
{
    return dy( g, g.bcy(), dir);
}
/**
 * @brief Create 3d derivative in z-direction
 *
 * @param g The grid on which to create dz (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A mpi matrix 
 */
RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH> dz( const MPI_Grid3d& g, direction dir = centered)
{
    return dz( g, g.bcz(), dir);
}
RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH> jumpY( const MPI_Grid2d& g)
{
    return jumpY( g, g.bcy());
}

RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH> jumpY( const MPI_Grid3d& g)
{
    return jumpY( g, g.bcy());
}
RowColDistMat<EllSparseBlockMat, CooSparseBlockMat, NNCH> jumpZ( const MPI_Grid3d& g)
{
    return jumpZ( g, g.bcz());
}




} //namespace create
} //namespace dg
