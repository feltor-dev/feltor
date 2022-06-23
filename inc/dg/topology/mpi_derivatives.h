#pragma once

#include "dg/backend/sparseblockmat.h"
#include "dg/backend/mpi_matrix.h"
#include "functions.h"
#include "derivatives.h"
#include "mpi_grid.h"

namespace dg{

namespace create{

///@cond
namespace detail{


/**
* @brief Iterate through elements of a matrix
*
* searches and stores all rows with elements with column -1 or num_rows in m,
* if there are no outer values
* returns an empty matrix and leaves input untouched. The rows in question
* are then replace by 0 in the input matrix.
* @param in The input matrix (contains only inner points on return)
*
* @return a newly created Coordinate matrix holding the outer rows
*/
template<class real_type>
CooSparseBlockMat<real_type> save_outer_values(EllSparseBlockMat<real_type>& in, const NNCH<real_type>& c)
{
    //search outer values in m
    CooSparseBlockMat<real_type> out( in.num_rows, 6, in.n, in.left_size, in.right_size);
    int index = in.data.size()/ in.n/in.n;
    thrust::host_vector<real_type> data_element(in.n*in.n, 0), zero(data_element);
    bool found=false;
    for( int i=0; i<in.num_rows; i++)
        for( int d=0; d<in.blocks_per_line; d++)
        {
            if( in.cols_idx[i*in.blocks_per_line+d]==-1)
            { //change the whole line
                for( int k=0; k<in.blocks_per_line; k++)
                {
                    for( int j=0; j<in.n*in.n; j++)
                        data_element[j] = in.data[ in.data_idx[i*in.blocks_per_line+k]*in.n*in.n + j];
                    int col = c.map_index( in.cols_idx[i*in.blocks_per_line+k]);
                    out.add_value( i, col, data_element);
                    in.data_idx[i*in.blocks_per_line+k] = index; //
                    in.cols_idx[i*in.blocks_per_line+k] = 0;
                }
                found=true;
            }
            if( in.cols_idx[i*in.blocks_per_line+d]==in.num_cols)
            {
                for( int k=0; k<in.blocks_per_line; k++)
                {
                    for( int j=0; j<in.n*in.n; j++)
                        data_element[j] = in.data[ in.data_idx[i*in.blocks_per_line+k]*in.n*in.n + j];
                    //assume col is either 3,4,or 5
                    int col = c.map_index( in.cols_idx[i*in.blocks_per_line+k]);
                    out.add_value( i, col, data_element);
                    in.data_idx[i*in.blocks_per_line+k] = index;
                    in.cols_idx[i*in.blocks_per_line+k] = in.num_cols-1;
                }
                found=true;
            }
        }
    if(found)
    {
        in.data.insert( in.data.end(), zero.begin(), zero.end());
    }

    //std::cout << "coo num entries "<<out.num_entries<<"\n";
    //for( int i=0; i<out.num_entries; i++)
    //    std::cout << "coo entries "<<out.cols_idx[i]<<" "<<out.data_idx[i]<<"\n";

    return out;
}

/**
* @brief Partition a global matrix into equal chunks among mpi processes
*
* grab the local rows of column and data indices and remap the column indices to vector with ghostcells
* copy the whole data array
* @param coord The mpi proces coordinate of the proper dimension
* @param howmany[3] # of processes 0 is left, 1 is the middle, 2 is right
* @return The reduced matrix
*/
template<class real_type>
EllSparseBlockMat<real_type> distribute_rows( const EllSparseBlockMat<real_type>& src, int coord, const int* howmany)
{
    if( howmany[1] == 1)
    {
        EllSparseBlockMat<real_type> temp(src);
        temp.set_left_size( temp.left_size/howmany[0]);
        temp.set_right_size( temp.right_size/howmany[2]);
        return temp;
    }
    assert( src.num_rows == src.num_cols);
    int chunk_size = src.num_rows/howmany[1];
    EllSparseBlockMat<real_type> temp(chunk_size, chunk_size, src.blocks_per_line, src.data.size()/(src.n*src.n), src.n);
    temp.set_left_size( src.left_size/howmany[0]);
    temp.set_right_size( src.right_size/howmany[2]);
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
template<class real_type>
RowColDistMat< EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> dx( const aRealMPITopology2d<real_type>& g, bc bcx, direction dir = centered)
{
    EllSparseBlockMat<real_type> matrix = dg::create::dx( g.global(), bcx, dir);
    unsigned vector_dimensions[] = {(unsigned)(g.nx()*g.local().Nx()), (unsigned)(g.ny()*g.local().Ny()), 1}; //x, y, z
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 2);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {dims[1], dims[0], 1}; //left, middle, right
    //distribute_rows, collective and save_outer_values are aware of howmany[1] == 1
    EllSparseBlockMat<real_type> inner = detail::distribute_rows(matrix, coords[0], howmany);
    NNCH<real_type> c( g.nx(), vector_dimensions, comm, 0);
    CooSparseBlockMat<real_type> outer = detail::save_outer_values(inner,c);

    return RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>>( inner, outer, c);
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
template<class real_type>
RowColDistMat< EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> dy( const aRealMPITopology2d<real_type>& g, bc bcy, direction dir = centered)
{
    EllSparseBlockMat<real_type> matrix = dg::create::dy( g.global(), bcy, dir);
    unsigned vector_dimensions[] = {(unsigned)(g.nx()*g.local().Nx()), (unsigned)(g.ny()*g.local().Ny()), 1}; //x, y, z
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 2);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {1, dims[1], dims[0]};
    EllSparseBlockMat<real_type> inner = detail::distribute_rows(matrix, coords[1], howmany);
    NNCH<real_type> c( g.ny(), vector_dimensions, comm, 1);
    CooSparseBlockMat<real_type> outer = detail::save_outer_values(inner,c);

    return RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>>( inner, outer, c);
}

/**
* @brief Create a 2d jump in the x-direction for mpi
*
* @param g A 2D mpi grid
* @param bcx boundary condition
*
* @return  A mpi matrix
*/
template<class real_type>
RowColDistMat< EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> jumpX( const aRealMPITopology2d<real_type>& g, bc bcx)
{
    EllSparseBlockMat<real_type> matrix = dg::create::jumpX( g.global(), bcx);
    unsigned vector_dimensions[] = {(unsigned)(g.nx()*g.local().Nx()), (unsigned)(g.ny()*g.local().Ny()), 1}; //x, y, z
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 2);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {dims[1], dims[0], 1}; //left, middle, right
    EllSparseBlockMat<real_type> inner = detail::distribute_rows(matrix, coords[0], howmany);
    NNCH<real_type> c( g.nx(), vector_dimensions, comm, 0);
    CooSparseBlockMat<real_type> outer = detail::save_outer_values(inner,c);

    return RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>>( inner, outer, c);
}
/**
* @brief Create a 2d jump in the y-direction for mpi
*
* @param g A 2D mpi grid
* @param bcy boundary condition
*
* @return  A mpi matrix
*/
template<class real_type>
RowColDistMat< EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> jumpY( const aRealMPITopology2d<real_type>& g, bc bcy)
{
    EllSparseBlockMat<real_type> matrix = dg::create::jumpY( g.global(), bcy);
    unsigned vector_dimensions[] = {(unsigned)(g.nx()*g.local().Nx()), (unsigned)(g.ny()*g.local().Ny()), 1}; //x, y, z
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 2);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {1, dims[1], dims[0]};
    EllSparseBlockMat<real_type> inner = detail::distribute_rows(matrix, coords[1], howmany);
    NNCH<real_type> c( g.ny(), vector_dimensions, comm, 1);
    CooSparseBlockMat<real_type> outer = detail::save_outer_values(inner,c);

    return RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>>( inner, outer, c);
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
template<class real_type>
RowColDistMat< EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> dx( const aRealMPITopology3d<real_type>& g, bc bcx, direction dir = centered)
{
    EllSparseBlockMat<real_type> matrix = dg::create::dx( g.global(), bcx, dir);
    unsigned vector_dimensions[] = {(unsigned)(g.nx()*g.local().Nx()), (unsigned)(g.ny()*g.local().Ny()), (unsigned)(g.nz()*g.local().Nz())}; //x, y, z
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {dims[2]*dims[1], dims[0], 1};
    EllSparseBlockMat<real_type> inner = detail::distribute_rows(matrix, coords[0], howmany);
    NNCH<real_type> c( g.nx(), vector_dimensions, comm, 0);
    CooSparseBlockMat<real_type> outer = detail::save_outer_values(inner,c);

    return RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>>( inner, outer, c);
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
template<class real_type>
RowColDistMat< EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> dy( const aRealMPITopology3d<real_type>& g, bc bcy, direction dir = centered)
{
    EllSparseBlockMat<real_type> matrix = dg::create::dy( g.global(), bcy, dir);
    unsigned vector_dimensions[] = {(unsigned)(g.nx()*g.local().Nx()), (unsigned)(g.ny()*g.local().Ny()), (unsigned)(g.nz()*g.local().Nz())}; //x, y, z
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {dims[2], dims[1], dims[0]};
    EllSparseBlockMat<real_type> inner = detail::distribute_rows(matrix, coords[1], howmany);
    NNCH<real_type> c( g.ny(), vector_dimensions, comm, 1);
    CooSparseBlockMat<real_type> outer = detail::save_outer_values(inner,c);

    return RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>>( inner, outer, c);
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
template<class real_type>
RowColDistMat< EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> dz( const aRealMPITopology3d<real_type>& g, bc bcz, direction dir = centered)
{
    EllSparseBlockMat<real_type> matrix = dg::create::dz( g.global(), bcz, dir);
    unsigned vector_dimensions[] = {(unsigned)(g.nx()*g.local().Nx()), (unsigned)(g.ny()*g.local().Ny()), (unsigned)(g.nz()*g.local().Nz())}; //x, y, z
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {1, dims[2], dims[1]*dims[0]};
    EllSparseBlockMat<real_type> inner = detail::distribute_rows(matrix, coords[2], howmany);
    NNCH<real_type> c( g.nz(), vector_dimensions, comm, 2);
    CooSparseBlockMat<real_type> outer = detail::save_outer_values(inner,c);

    return RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>>( inner, outer, c);
}

/**
* @brief Create a 3d jump in the x-direction for mpi
*
* @param g A 3D mpi grid
* @param bcx boundary condition
*
* @return  A mpi matrix
*/
template<class real_type>
RowColDistMat< EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> jumpX( const aRealMPITopology3d<real_type>& g, bc bcx)
{
    EllSparseBlockMat<real_type> matrix = dg::create::jumpX( g.global(), bcx);
    unsigned vector_dimensions[] = {(unsigned)(g.nx()*g.local().Nx()), (unsigned)(g.ny()*g.local().Ny()), (unsigned)(g.nz()*g.local().Nz())}; //x, y, z
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {dims[2]*dims[1], dims[0], 1};
    EllSparseBlockMat<real_type> inner = detail::distribute_rows(matrix, coords[0], howmany);
    NNCH<real_type> c( g.nx(), vector_dimensions, comm, 0);
    CooSparseBlockMat<real_type> outer = detail::save_outer_values(inner,c);

    return RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>>( inner, outer, c);
}

/**
* @brief Create a 3d jump in the y-direction for mpi
*
* @param g A 3D mpi grid
* @param bcy boundary condition
*
* @return  A mpi matrix
*/
template<class real_type>
RowColDistMat< EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> jumpY( const aRealMPITopology3d<real_type>& g, bc bcy)
{
    EllSparseBlockMat<real_type> matrix = dg::create::jumpY( g.global(), bcy);
    unsigned vector_dimensions[] = {(unsigned)(g.nx()*g.local().Nx()), (unsigned)(g.ny()*g.local().Ny()), (unsigned)(g.nz()*g.local().Nz())}; //x, y, z
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {dims[2], dims[1], dims[0]};
    EllSparseBlockMat<real_type> inner = detail::distribute_rows(matrix, coords[1], howmany);
    NNCH<real_type> c( g.ny(), vector_dimensions, comm, 1);
    CooSparseBlockMat<real_type> outer = detail::save_outer_values(inner,c);

    return RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>>( inner, outer, c);
}
/**
* @brief Create a 3d jump in the z-direction for mpi
*
* @param g A 3D mpi grid
* @param bcz boundary condition
*
* @return  A mpi matrix
*/
template<class real_type>
RowColDistMat< EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> jumpZ( const aRealMPITopology3d<real_type>& g, bc bcz)
{
    EllSparseBlockMat<real_type> matrix = dg::create::jumpZ( g.global(), bcz);
    unsigned vector_dimensions[] = {(unsigned)(g.nx()*g.local().Nx()), (unsigned)(g.ny()*g.local().Ny()), (unsigned)(g.nz()*g.local().Nz())}; //x, y, z
    MPI_Comm comm = g.communicator();
    int ndims;
    MPI_Cartdim_get( comm, &ndims);
    assert( ndims == 3);
    int dims[ndims], periods[ndims], coords[ndims];
    MPI_Cart_get( comm, ndims, dims, periods, coords);

    int howmany[] = {1, dims[2], dims[1]*dims[0]};
    EllSparseBlockMat<real_type> inner = detail::distribute_rows(matrix, coords[2], howmany);
    NNCH<real_type> c( g.nz(), vector_dimensions, comm, 2);
    CooSparseBlockMat<real_type> outer = detail::save_outer_values(inner,c);

    return RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>>( inner, outer, c);
}

/**
 * @brief Create 2d derivative in x-direction
 *
 * @param g The grid on which to create dx (boundary condition is taken from here)
 * @param dir The direction of the first derivative
 *
 * @return A mpi matrix
 */
template<class real_type>
RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> dx( const aRealMPITopology2d<real_type>& g, direction dir = centered)
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
template<class real_type>
RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> dx( const aRealMPITopology3d<real_type>& g, direction dir = centered)
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
template<class real_type>
RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> jumpX( const aRealMPITopology2d<real_type>& g)
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
template<class real_type>
RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> jumpX( const aRealMPITopology3d<real_type>& g)
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
template<class real_type>
RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> dy( const aRealMPITopology2d<real_type>& g, direction dir = centered)
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
template<class real_type>
RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> dy( const aRealMPITopology3d<real_type>& g, direction dir = centered)
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
template<class real_type>
RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> jumpY( const aRealMPITopology2d<real_type>& g)
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
template<class real_type>
RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> jumpY( const aRealMPITopology3d<real_type>& g)
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
template<class real_type>
RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> dz( const aRealMPITopology3d<real_type>& g, direction dir = centered)
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
template<class real_type>
RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> jumpZ( const aRealMPITopology3d<real_type>& g)
{
    return jumpZ( g, g.bcz());
}


///@}


} //namespace create
} //namespace dg
