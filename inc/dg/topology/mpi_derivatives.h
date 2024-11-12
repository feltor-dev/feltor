#pragma once

#include "dg/backend/sparseblockmat.h"
#include "dg/backend/mpi_matrix.h"
#include "dg/backend/typedefs.h"
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
* are then replaced by 0 in the input matrix.
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
    // TODO should we unify directions of kronecker "left" and "right"
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
//


/**
* @brief Create a derivative along the given coordinate for mpi
*
* @param coord the coordinate along which to derive
* @param g A mpi grid
* @param bc boundary condition
* @param dir centered, forward or backward
*
* @return  A mpi matrix
*/
template<class real_type, size_t Nd>
dg::MHMatrix_t<real_type> derivative( unsigned coord,
    const aRealMPITopology<real_type, Nd>& g, dg::bc bc, direction dir = centered)
{
    EllSparseBlockMat<real_type> matrix = dg::create::derivative( coord,
            g.global(), bc, dir);
    int dims[Nd], periods[Nd], coords[Nd];
    MPI_Cart_get( g.communicator(), Nd, dims, periods, coords);

    int howmany[3] = {1, dims[coord], 1}; // left, middle, right
    for( unsigned u=0; u<coord; u++)
        howmany[2]*= dims[u];
    for( unsigned u=coord+1; u<Nd; u++)
        howmany[0]*= dims[u];
    EllSparseBlockMat<real_type> inner = detail::distribute_rows(matrix,
            coords[coord], howmany);
    NNCH<real_type> c( g.n(coord), g.local().get_shape(), g.communicator(), coord);
    CooSparseBlockMat<real_type> outer = detail::save_outer_values(inner,c);

    return { inner, outer, c};
}

/**
 * @brief Create a jump matrix along given coordinate for mpi
 *
 * @param coord the coordinate along which to jump
 * @param g The mpi grid on which to create jump
 * @param bc The boundary condition
 *
 * @return A mpi matrix
 */
template<class real_type, size_t Nd>
dg::MHMatrix_t<real_type> jump( unsigned coord,
    const aRealMPITopology<real_type, Nd>& g, dg::bc bc)
{
    EllSparseBlockMat<real_type> matrix = dg::create::jump( coord,
            g.global(), bc);
    int dims[Nd], periods[Nd], coords[Nd];
    MPI_Cart_get( g.communicator(), Nd, dims, periods, coords);

    int howmany[3] = {1, dims[coord], 1}; // left, middle, right
    for( unsigned u=0; u<coord; u++)
        howmany[2]*= dims[u];
    for( unsigned u=coord+1; u<Nd; u++)
        howmany[0]*= dims[u];
    EllSparseBlockMat<real_type> inner = detail::distribute_rows(matrix,
            coords[coord], howmany);
    NNCH<real_type> c( g.n(coord), g.local().get_shape(), g.communicator(), coord);
    CooSparseBlockMat<real_type> outer = detail::save_outer_values(inner,c);

    return { inner, outer, c};
}

///@}


} //namespace create
} //namespace dg
