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

    return out;
}

//TODO update docu
/**
* @brief Partition a global matrix into equal chunks among mpi processes
*
* grab the local rows of column and data indices and remap the column indices to vector with ghostcells
* copy the whole data array
* @param coord The mpi proces coordinate in the derived dimension
* @param howmany # of processes in the derived dimension
* @return The reduced matrix
*/
template<class real_type>
EllSparseBlockMat<real_type> distribute( const EllSparseBlockMat<real_type>& src,
    const aRealMPITopology<real_type,1>& g_rows, const aRealMPITopology<real_type,1>& g_cols, bool& isCommunicating)
{
    // src left_size and src right_size are the correct local sizes
    // src has global rows and global cols
    // grid is needed for local2globalIdx
    // communicators need to be the same
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    int dim, period, coord;
    MPI_Cart_get( g_cols.communicator(), 1, &dim, &period, &coord); // coord of the calling process
    if( dim == 1)
    {
        return src;
    }
    EllSparseBlockMat<real_type> temp(g_rows.local().N(), g_cols.local().N(), // number of blocks
        src.blocks_per_line, src.data.size()/(src.n*src.n), src.n);
    temp.set_left_size( src.left_size);
    temp.set_right_size( src.right_size);
    std::cout << "rows cols "<<g_rows.local().N()<<" "<<g_cols.local().N()<<"\n";
    std::cout << "temp "<<temp.num_rows<<" "<<temp.num_cols<<"\n";
    std::cout <<std::endl;
    //first copy data elements (even though not all might be needed it doesn't slow down things either)
    for( unsigned  i=0; i<src.data.size(); i++)
        temp.data[i] = src.data[i];
    //now grab the right rows of the cols and data indices
    if(rank==0) std::cout << "Global\n";
    for( unsigned i=0; i<(unsigned)temp.num_rows; i++)
    for( unsigned k=0; k<(unsigned)temp.blocks_per_line; k++)
    {
        int gIdx=0;
        int bpl = temp.blocks_per_line;
        if(rank==0)std::cout << src.data_idx[i*bpl + k] << " ";
        if(rank==0)std::cout << src.cols_idx[i*bpl + k] << " ";
        g_rows.local2globalIdx( i*src.n, coord, gIdx); // convert from idx to block idx !
        temp.data_idx[i*bpl + k] = src.data_idx[ gIdx/src.n*bpl + k];
        temp.cols_idx[i*bpl + k] = src.cols_idx[ gIdx/src.n*bpl + k];
        if(rank==0)std::cout << temp.data_idx[i*bpl + k] << " ";
        if(rank==0)std::cout << temp.cols_idx[i*bpl + k] << "\n";
    }
    if(rank==0)std::cout << std::endl;
    bool local_communicating = false;
    isCommunicating = false;
    //data indices are correct but cols still contain global indices (remapping a bit clumsy)
    for( unsigned i=0; i<temp.cols_idx.size(); i++)
    {
        //first in the zeroth line the col idx might be (global)num_cols - 1 -> map that to -1
        if( coord==0 and i/src.blocks_per_line == 0 and temp.cols_idx[i] == src.num_cols-1)
        {
            temp.cols_idx[i] = -1;
            local_communicating = true;
        }
        //second in the last line the col idx might be 0 -> map to (global)num_cols
        else if( coord==(dim-1) and (int)i/src.blocks_per_line == temp.num_rows-1 and temp.cols_idx[i] == 0)
        {
            temp.cols_idx[i] = g_cols.local().N();
            local_communicating = true;
        }
        else
        {
            int global_start_idx  = 0;
            g_cols.local2globalIdx( 0, coord, global_start_idx);
            global_start_idx/=src.n; //convert to block idx!
            // shift to range [-1, ..., local_shape]
            int shifted = temp.cols_idx[i] - global_start_idx;
            if( shifted < -1 or shifted > (int)g_cols.local().N())
                std::cerr << "Idx "<<i<<" "<<temp.cols_idx[i]<<" shifted "<<shifted
                    <<" local shape "<<g_cols.local().N()<<" "<<global_start_idx<<"\n";
            assert( shifted >= -1 and shifted <= (int)g_cols.local().N());
            temp.cols_idx[i] = shifted;
            if( temp.cols_idx[i] == -1)
                local_communicating = true;
            if( (unsigned)temp.cols_idx[i] == g_cols.local().N())
                local_communicating = true;
        }
    }
    // Everyone needs to agree on whether we communicate
    MPI_Allreduce( &local_communicating, &isCommunicating, 1,
                   MPI_C_BOOL, MPI_LOR, g_cols.communicator());

    return temp;
}

// Also used by fast_interpolation
//
template<class real_type, size_t Nd>
RealGrid<real_type,Nd> local_global_grid( unsigned coord,
    const aRealMPITopology<real_type, Nd>& g)
{
    // global grid in coord, local grids else
    std::array<RealGrid<real_type,1>,Nd> axes;
    for( unsigned u=0; u<Nd; u++)
        axes[u] = g.local().axis(u);
    axes[coord] = g.global().axis(coord);
    return RealGrid<real_type,Nd>{axes};
}
template<class real_type, size_t Nd>
dg::MHMatrix_t<real_type> elevate( unsigned coord, const dg::HMatrix_t<real_type>& matrix, const aRealMPITopology<real_type, Nd>& g_rows, const aRealMPITopology<real_type,Nd>& g_cols)
{
    // TODO Communicators must be the same
    bool isCommunicating = true;
    EllSparseBlockMat<real_type> inner = detail::distribute(matrix,
            g_rows.axis(coord), g_cols.axis(coord), isCommunicating);
    if( isCommunicating)
    {
        NNCH<real_type> c( g_cols.n(coord), g_cols.local().get_shape(), g_cols.communicator(), coord);
        CooSparseBlockMat<real_type> outer = detail::save_outer_values(inner,c);

        return { inner, outer, c};
    }
    else
    {
        return { inner, CooSparseBlockMat<real_type>(), NNCH<real_type>(g_cols.communicator())};
    }
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
    return detail::elevate( coord, dg::create::derivative( coord,
            detail::local_global_grid(coord, g), bc, dir), g, g);
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
    return detail::elevate( coord, dg::create::jump( coord,
            detail::local_global_grid(coord, g), bc), g, g);
}

///@}


} //namespace create
} //namespace dg
