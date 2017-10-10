#pragma once

#include "cusp_matrix_blas.cuh"
#include "mpi_grid.h"
#include "mpi_matrix.h"
#include "mpi_vector.h"
#include "mpi_collective.h"
#include "projection.cuh"
#include "typedefs.cuh"

/*! @file
  @brief Useful MPI typedefs and overloads of interpolation and projection
  */

namespace dg
{
///@addtogroup typedefs
///@{
//interpolation matrices
typedef MPIDistMat< dg::IHMatrix, GeneralComm< dg::iHVec, dg::HVec > > MIHMatrix; //!< MPI distributed CSR host Matrix
typedef MPIDistMat< dg::IDMatrix, GeneralComm< dg::iDVec, dg::DVec > > MIDMatrix; //!< MPI distributed CSR device Matrix

namespace detail{
//given global indices -> make a sorted unique indices vector -> make a gather map into the unique vector
//buffer_idx -> (gather map/ new column indices) same size as global_idx ( can alias global_idx, index into unique_global_idx
//unique_global_idx -> (list of unique global indices to be used in a Collective Communication object)
void global2bufferIdx( const cusp::array1d<int, cusp::host_memory>& global_idx, cusp::array1d<int, cusp::host_memory>& buffer_idx, thrust::host_vector<int>& locally_unique_global_idx)
{
    thrust::host_vector<int> index(global_idx.begin(), global_idx.end()), m_global_idx(index);
    thrust::sequence( index.begin(), index.end());
    //1. sort input global indices
    thrust::stable_sort_by_key( m_global_idx.begin(), m_global_idx.end(), index.begin());//this changes both global_idx and index
    //2. now reduce on multiple indices
    thrust::host_vector<int> ones( index.size(), 1);
    thrust::host_vector<int> unique_global( index.size()), howmany( index.size());
    typedef typename thrust::host_vector<int>::iterator iterator;
    thrust::pair<iterator, iterator> new_end;
    new_end = thrust::reduce_by_key( m_global_idx.begin(), m_global_idx.end(), ones.begin(), unique_global.begin(), howmany.begin());
    //3. copy unique indices
    locally_unique_global_idx.assign( unique_global.begin(), new_end.first);
    //4. manually make gather map into locally_unique_global_idx
    thrust::host_vector<int> gather_map;
    for( int i=0; i<(int)locally_unique_global_idx.size(); i++)
        for( int j=0; j<howmany[i]; j++)
            gather_map.push_back(i);
    assert( gather_map.size() == global_idx.size());
    //5. buffer idx is the new index 
    buffer_idx.resize( global_idx.size());
    thrust::scatter( gather_map.begin(), gather_map.end(), index.begin(), buffer_idx.begin());
}
}//namespace detail
///@}

/**
 * @brief Convert a matrix with local row and global column indices to a row distributed MPI matrix
 *
 * Checks if communication is actually needed
 * @param global the column indices need to be global, the row indices local
 * @param topology the topology defines how the indices are converted from global to local
 *
 * @return a row distributed MPI matrix
 * @ingroup misc
 */
dg::MIHMatrix convert_row_dist( const dg::IHMatrix& global, const aMPITopology2d& topology) 
{
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    dg::iHVec unique_global_idx;
    cusp::array1d<int, cusp::host_memory> buffer_idx;
    dg::detail::global2bufferIdx( global.column_indices, buffer_idx, unique_global_idx);
    dg::GeneralComm<dg::iHVec, dg::HVec> comm( unique_global_idx, topology);
    dg::IHMatrix local( global.num_rows, comm.size(), global.values.size());
    local.row_offsets=global.row_offsets;
    local.column_indices=buffer_idx;
    local.values=global.values;
    if( !comm.isCommunicating() ) 
    {
        cusp::array1d<int, cusp::host_memory> local_idx(global.column_indices), pids(local_idx);
        bool success = true;
        for(unsigned i=0; i<local_idx.size(); i++)
            if( !topology.global2localIdx(global.column_indices[i], local_idx[i], pids[i]) ) success = false;
        assert( success);
        local.column_indices=local_idx;
        comm = dg::GeneralComm< dg::iHVec, dg::HVec>();
    }
    dg::MIHMatrix matrix(   local, comm, dg::row_dist);
    return matrix;
}

namespace create
{

///@addtogroup interpolation
///@{

///@copydoc dg::create::interpolation(const Grid1d&,const Grid1d&)
dg::MIHMatrix interpolation( const aMPITopology2d& g_new, const aMPITopology2d& g_old)
{
    return MIHMatrix( interpolation( g_new.local(), g_old.local()), GeneralComm<iHVec, HVec>());
}
///@copydoc interpolation(const Grid1d&,const Grid1d&)
dg::MIHMatrix interpolation( const aMPITopology3d& g_new, const aMPITopology3d& g_old)
{
    return MIHMatrix( interpolation( g_new.local(), g_old.local()), GeneralComm<iHVec, HVec>());
}

///@copydoc interpolationT(const Grid1d&,const Grid1d&)
dg::MIHMatrix interpolationT( const aMPITopology2d& g_new, const aMPITopology2d& g_old)
{
    return MIHMatrix( interpolationT( g_new.local(), g_old.local()), GeneralComm<iHVec, HVec>());
}
///@copydoc interpolationT(const Grid1d&,const Grid1d&)
dg::MIHMatrix interpolationT( const aMPITopology3d& g_new, const aMPITopology3d& g_old)
{
    return MIHMatrix( interpolationT( g_new.local(), g_old.local()), GeneralComm<iHVec, HVec>());
}

///@copydoc projection(const Grid1d&,const Grid1d&)
dg::MIHMatrix projection( const aMPITopology2d& g_new, const aMPITopology2d& g_old)
{
    return MIHMatrix( projection( g_new.local(), g_old.local()), GeneralComm<iHVec, HVec>());
}
///@copydoc projection(const Grid1d&,const Grid1d&)
dg::MIHMatrix projection( const aMPITopology3d& g_new, const aMPITopology3d& g_old)
{
    return MIHMatrix( projection( g_new.local(), g_old.local()), GeneralComm<iHVec, HVec>());
}

/**
 * @brief Create an MPI row distributed interpolation matrix 
 *
 * @copydetails interpolation(const thrust::host_vector<double>&,const thrust::host_vector<double>&,const aTopology2d&,dg::bc,dg::bc)
 */
dg::MIHMatrix interpolation( const dg::HVec& x, const dg::HVec& y, const aMPITopology2d& grid, dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU)
{
    return convert_row_dist(  
            dg::create::interpolation( x, y, grid.global(), bcx, bcy), 
            grid);
}



///@}

}//namespace create
}//namespace dg
