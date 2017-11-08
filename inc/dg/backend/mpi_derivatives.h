#pragma once

#include "functions.h"
#include "sparseblockmat.h"
#include "derivatives.h"
#include "mpi_vector.h"
#include "mpi_matrix.h"

namespace dg{

namespace create{ 

///@cond
namespace detail{ 


/**
* @brief Iterate through elements of a matrix
*
* searches and stores all elements with column -1 or num_rows in m, 
* if there are no outer values
* returns an empty matrix and leaves m untouched
* @param m The input matrix (contains only inner points on return)
*
* @return a newly created Coordinate matrix holding the outer points
*/
CooSparseBlockMat<double> save_outer_values(EllSparseBlockMat<double>& m)
{
    //search outer values in m
    CooSparseBlockMat<double> mat( m.num_rows, 2, m.n, m.left_size, m.right_size);
    int index = m.data.size()/ m.n/m.n;
    thrust::host_vector<double> data_element(m.n*m.n, 0), zero(data_element);
    bool found=false;
    for( int i=0; i<m.num_rows; i++)
        for( int d=0; d<m.blocks_per_line; d++)
        {
            if( m.cols_idx[i*m.blocks_per_line+d]==-1)
            {
                for( int j=0; j<m.n*m.n; j++)
                    data_element[j] = m.data[ m.data_idx[i*m.blocks_per_line+d]*m.n*m.n + j];
                mat.add_value( i, 0, data_element);
                m.data_idx[i*m.blocks_per_line+d] = index; //
                m.cols_idx[i*m.blocks_per_line+d] = 0;
                found=true;
            }
            if( m.cols_idx[i*m.blocks_per_line+d]==m.num_cols)
            {
                for( int j=0; j<m.n*m.n; j++)
                    data_element[j] = m.data[ m.data_idx[i*m.blocks_per_line+d]*m.n*m.n + j];
                mat.add_value( i, 1, data_element);
                m.data_idx[i*m.blocks_per_line+d] = index;
                m.cols_idx[i*m.blocks_per_line+d] = m.num_cols-1;
                found=true;
            }
        }
    if(found)
        m.data.insert( m.data.end(), zero.begin(), zero.end()); 
    return mat;
}

/**
* @brief Reduce a global matrix into equal chunks among mpi processes
*
* grabs the right chunk of column and data indices and remaps the column indices to vector with ghostcells
* copies the whole data array 
* @param coord The mpi proces coordinate of the proper dimension
* @param howmany[3] # of processes 0 is left, 1 is the middle, 2 is right
* @return The reduced matrix
*/
EllSparseBlockMat<double> distribute_rows( const EllSparseBlockMat<double>& src, int coord, const int* howmany)
{
    if( howmany[1] == 1)
    {
        EllSparseBlockMat<double> temp(src);
        temp.left_size = temp.left_size/howmany[0];
        temp.right_size = temp.right_size/howmany[2];
        temp.set_default_range();
        return temp;
    }
    assert( src.num_rows == src.num_cols);
    int chunk_size = src.num_rows/howmany[1];
    EllSparseBlockMat<double> temp(chunk_size, chunk_size, src.blocks_per_line, src.data.size()/(src.n*src.n), src.n);
    temp.left_size = src.left_size/howmany[0];
    temp.right_size = src.right_size/howmany[2];
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
        if( coord==0 && i/src.blocks_per_line == 0 && temp.cols_idx[i] == src.num_cols-1) temp.cols_idx[i] = -1; 
        //second in the last line the col idx mighty be 0 -> map to (global)num_cols
        if( coord==(howmany[1]-1)&& (int)i/src.blocks_per_line == temp.num_rows-1 && temp.cols_idx[i] == 0) temp.cols_idx[i] = src.num_cols;  
        //Elements are now in the range -1, 0, 1,..., (global)num_cols
        //now shift this range to chunk range -1,..,chunk_size
        temp.cols_idx[i] = (temp.cols_idx[i] - coord*chunk_size ); 
    }
    temp.set_default_range();
    return temp;
}


} //namespace detail

///@endcond
///@addtogroup creation
///@{

/**
* @brief Create a 2d derivative in the x-direction for mpi
*
* @param g A 2D mpi grid
* @param bcx boundary condition
* @param dir centered, forward or backward
*
* @return  A mpi matrix
*/
RowColDistMat< EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> dx( const aMPITopology2d& g, bc bcx, direction dir = centered)
{
    EllSparseBlockMat<double> matrix = dg::create::dx( g.global(), bcx, dir);
    unsigned vector_dimensions[] = {(unsigned)(g.n()*g.Nx()), (unsigned)(g.n()*g.Ny()), 1}; //x, y, z
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 2);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {dims[1], dims[0], 1}; //left, middle, right
    //distribute_rows, collective and save_outer_values are aware of howmany[1] == 1
    EllSparseBlockMat<double> inner = detail::distribute_rows(matrix, coords[0], howmany);
    NNCH c( g.n(), vector_dimensions, comm, 0);
    CooSparseBlockMat<double> outer = detail::save_outer_values(inner);

    return RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH>( inner, outer, c);
}

/**
* @brief Create a 2d derivative in the y-direction for mpi
*
* @param g A 2D mpi grid
* @param bcy boundary condition
* @param dir centered, forward or backward
*
* @return  A mpi matrix
*/
RowColDistMat< EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> dy( const aMPITopology2d& g, bc bcy, direction dir = centered)
{
    EllSparseBlockMat<double> matrix = dg::create::dy( g.global(), bcy, dir);
    unsigned vector_dimensions[] = {(unsigned)(g.n()*g.Nx()), (unsigned)(g.n()*g.Ny()), 1}; //x, y, z
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 2);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {1, dims[1], dims[0]};
    EllSparseBlockMat<double> inner = detail::distribute_rows(matrix, coords[1], howmany);
    NNCH c( g.n(), vector_dimensions, comm, 1);
    CooSparseBlockMat<double> outer = detail::save_outer_values(inner);

    return RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH>( inner, outer, c);
}

/**
* @brief Create a 2d jump in the x-direction for mpi
*
* @param g A 2D mpi grid
* @param bcx boundary condition
*
* @return  A mpi matrix
*/
RowColDistMat< EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> jumpX( const aMPITopology2d& g, bc bcx)
{
    EllSparseBlockMat<double> matrix = dg::create::jumpX( g.global(), bcx);
    unsigned vector_dimensions[] = {(unsigned)(g.n()*g.Nx()), (unsigned)(g.n()*g.Ny()), 1}; //x, y, z
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 2);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {dims[1], dims[0], 1}; //left, middle, right
    EllSparseBlockMat<double> inner = detail::distribute_rows(matrix, coords[0], howmany);
    NNCH c( g.n(), vector_dimensions, comm, 0);
    CooSparseBlockMat<double> outer = detail::save_outer_values(inner);

    return RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH>( inner, outer, c);
}
/**
* @brief Create a 2d jump in the y-direction for mpi
*
* @param g A 2D mpi grid
* @param bcy boundary condition
*
* @return  A mpi matrix
*/
RowColDistMat< EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> jumpY( const aMPITopology2d& g, bc bcy)
{
    EllSparseBlockMat<double> matrix = dg::create::jumpY( g.global(), bcy);
    unsigned vector_dimensions[] = {(unsigned)(g.n()*g.Nx()), (unsigned)(g.n()*g.Ny()), 1}; //x, y, z
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 2);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {1, dims[1], dims[0]};
    EllSparseBlockMat<double> inner = detail::distribute_rows(matrix, coords[1], howmany);
    NNCH c( g.n(), vector_dimensions, comm, 1);
    CooSparseBlockMat<double> outer = detail::save_outer_values(inner);

    return RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH>( inner, outer, c);
}

/**
* @brief Create a 3d derivative in the x-direction for mpi
*
* @param g A 3D mpi grid
* @param bcx boundary condition
* @param dir centered, forward or backward
*
* @return  A mpi matrix
*/
RowColDistMat< EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> dx( const aMPITopology3d& g, bc bcx, direction dir = centered)
{
    EllSparseBlockMat<double> matrix = dg::create::dx( g.global(), bcx, dir);
    unsigned vector_dimensions[] = {(unsigned)(g.n()*g.Nx()), (unsigned)(g.n()*g.Ny()), (unsigned)(g.Nz())}; //x, y, z
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {dims[2]*dims[1], dims[0], 1};
    EllSparseBlockMat<double> inner = detail::distribute_rows(matrix, coords[0], howmany);
    NNCH c( g.n(), vector_dimensions, comm, 0);
    CooSparseBlockMat<double> outer = detail::save_outer_values(inner);

    return RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH>( inner, outer, c);
}
/**
* @brief Create a 3d derivative in the y-direction for mpi
*
* @param g A 3D mpi grid
* @param bcy boundary condition
* @param dir centered, forward or backward
*
* @return  A mpi matrix
*/
RowColDistMat< EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> dy( const aMPITopology3d& g, bc bcy, direction dir = centered)
{
    EllSparseBlockMat<double> matrix = dg::create::dy( g.global(), bcy, dir);
    unsigned vector_dimensions[] = {(unsigned)(g.n()*g.Nx()), (unsigned)(g.n()*g.Ny()), (unsigned)(g.Nz())}; //x, y, z
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {dims[2], dims[1], dims[0]};
    EllSparseBlockMat<double> inner = detail::distribute_rows(matrix, coords[1], howmany);
    NNCH c( g.n(), vector_dimensions, comm, 1);
    CooSparseBlockMat<double> outer = detail::save_outer_values(inner);

    return RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH>( inner, outer, c);
}
/**
* @brief Create a 3d derivative in the z-direction for mpi
*
* @param g A 3D mpi grid
* @param bcz boundary condition
* @param dir centered, forward or backward
*
* @return  A mpi matrix
*/
RowColDistMat< EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> dz( const aMPITopology3d& g, bc bcz, direction dir = centered)
{
    EllSparseBlockMat<double> matrix = dg::create::dz( g.global(), bcz, dir);
    unsigned vector_dimensions[] = {(unsigned)(g.n()*g.Nx()), (unsigned)(g.n()*g.Ny()), (unsigned)(g.Nz())}; //x, y, z
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {1, dims[2], dims[1]*dims[0]};
    EllSparseBlockMat<double> inner = detail::distribute_rows(matrix, coords[2], howmany);
    NNCH c( 1, vector_dimensions, comm, 2);
    CooSparseBlockMat<double> outer = detail::save_outer_values(inner);

    return RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH>( inner, outer, c);
}

/**
* @brief Create a 3d jump in the x-direction for mpi
*
* @param g A 3D mpi grid
* @param bcx boundary condition
*
* @return  A mpi matrix
*/
RowColDistMat< EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> jumpX( const aMPITopology3d& g, bc bcx)
{
    EllSparseBlockMat<double> matrix = dg::create::jumpX( g.global(), bcx);
    unsigned vector_dimensions[] = {(unsigned)(g.n()*g.Nx()), (unsigned)(g.n()*g.Ny()), (unsigned)(g.Nz())}; //x, y, z
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {dims[2]*dims[1], dims[0], 1};
    EllSparseBlockMat<double> inner = detail::distribute_rows(matrix, coords[0], howmany);
    NNCH c( g.n(), vector_dimensions, comm, 0);
    CooSparseBlockMat<double> outer = detail::save_outer_values(inner);

    return RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH>( inner, outer, c);
}

/**
* @brief Create a 3d jump in the y-direction for mpi
*
* @param g A 3D mpi grid
* @param bcy boundary condition
*
* @return  A mpi matrix
*/
RowColDistMat< EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> jumpY( const aMPITopology3d& g, bc bcy)
{
    EllSparseBlockMat<double> matrix = dg::create::jumpY( g.global(), bcy);
    unsigned vector_dimensions[] = {(unsigned)(g.n()*g.Nx()), (unsigned)(g.n()*g.Ny()), (unsigned)(g.Nz())}; //x, y, z
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {dims[2], dims[1], dims[0]};
    EllSparseBlockMat<double> inner = detail::distribute_rows(matrix, coords[1], howmany);
    NNCH c( g.n(), vector_dimensions, comm, 1);
    CooSparseBlockMat<double> outer = detail::save_outer_values(inner);

    return RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH>( inner, outer, c);
}
/**
* @brief Create a 3d jump in the z-direction for mpi
*
* @param g A 3D mpi grid
* @param bcz boundary condition
*
* @return  A mpi matrix
*/
RowColDistMat< EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> jumpZ( const aMPITopology3d& g, bc bcz)
{
    EllSparseBlockMat<double> matrix = dg::create::jumpZ( g.global(), bcz);
    unsigned vector_dimensions[] = {(unsigned)(g.n()*g.Nx()), (unsigned)(g.n()*g.Ny()), (unsigned)(g.Nz())}; //x, y, z
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {1, dims[2], dims[1]*dims[0]};
    EllSparseBlockMat<double> inner = detail::distribute_rows(matrix, coords[2], howmany);
    NNCH c( 1, vector_dimensions, comm, 2);
    CooSparseBlockMat<double> outer = detail::save_outer_values(inner);

    return RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH>( inner, outer, c);
}

/**
 * @brief Create 2d derivative in x-direction
 *
 * @param g The grid on which to create dx (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A mpi matrix 
 */
RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> dx( const aMPITopology2d& g, direction dir = centered)
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
RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> dx( const aMPITopology3d& g, direction dir = centered)
{
    return dx( g, g.bcx(), dir);
}
/**
 * @brief Create 2d jump in x-direction
 *
 * @param g The grid on which to create jump (boundary condition is taken from here)
 *
 * @return A mpi matrix 
 */
RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> jumpX( const aMPITopology2d& g)
{
    return jumpX( g, g.bcx());
}

/**
 * @brief Create 3d jump in x-direction
 *
 * @param g The grid on which to create jump (boundary condition is taken from here)
 *
 * @return A mpi matrix 
 */
RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> jumpX( const aMPITopology3d& g)
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
RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> dy( const aMPITopology2d& g, direction dir = centered)
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
RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> dy( const aMPITopology3d& g, direction dir = centered)
{
    return dy( g, g.bcy(), dir);
}

/**
 * @brief Create 2d jump in y-direction
 *
 * @param g The grid on which to create dy (boundary condition is taken from here)
 *
 * @return A mpi matrix
 */
RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> jumpY( const aMPITopology2d& g)
{
    return jumpY( g, g.bcy());
}

/**
 * @brief Create 3d jump in y-direction
 *
 * @param g The grid on which to create dy (boundary condition is taken from here)
 *
 * @return A mpi matrix 
 */
RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> jumpY( const aMPITopology3d& g)
{
    return jumpY( g, g.bcy());
}

/**
 * @brief Create 3d derivative in z-direction
 *
 * @param g The grid on which to create dz (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A mpi matrix 
 */
RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> dz( const aMPITopology3d& g, direction dir = centered)
{
    return dz( g, g.bcz(), dir);
}

/**
 * @brief Create 3d jump in z-direction
 *
 * @param g The grid on which to create dz (boundary condition is taken from here)
 *
 * @return A mpi matrix 
 */
RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> jumpZ( const aMPITopology3d& g)
{
    return jumpZ( g, g.bcz());
}


///@} 


} //namespace create
} //namespace dg
