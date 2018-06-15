#pragma once

#include "dg/backend/typedefs.cuh"
#include "dg/backend/mpi_matrix.h"
#include "dg/backend/mpi_collective.h"
#include "mpi_grid.h"
#include "projection.h"

/*! @file
  @brief Useful MPI typedefs and overloads of interpolation and projection
  */

namespace dg
{
///@addtogroup typedefs
///@{
template<class real_type>
using tMIHMatrix = MPIDistMat< tIHMatrix<real_type>, GeneralComm< dg::iHVec, thrust::host_vector<real_type>> >;
template<class real_type>
using tMIDMatrix = MPIDistMat< tIDMatrix<real_type>, GeneralComm< dg::iDVec, thrust::device_vector<real_type>> >;
using MIHMatrix = tMIHMatrix<double>;
using MIDMatrix = tMIDMatrix<double>;
//typedef MPIDistMat< dg::IHMatrix, GeneralComm< dg::iHVec, dg::HVec > > MIHMatrix; //!< MPI distributed CSR host Matrix
//typedef MPIDistMat< dg::IDMatrix, GeneralComm< dg::iDVec, dg::DVec > > MIDMatrix; //!< MPI distributed CSR device Matrix
///@}

///@cond
namespace detail{
//given global indices -> make a sorted unique indices vector -> make a gather map into the unique vector
//buffer_idx -> (gather map/ new column indices) same size as global_idx ( can alias global_idx, index into unique_global_idx
//unique_global_idx -> (list of unique global indices to be used in a Collective Communication object)
static void global2bufferIdx( const cusp::array1d<int, cusp::host_memory>& global_idx, cusp::array1d<int, cusp::host_memory>& buffer_idx, thrust::host_vector<int>& locally_unique_global_idx)
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
///@endcond

/**
 * @brief Convert a matrix with local row and global column indices to a row distributed MPI matrix
 *
 * @tparam ConversionPolicy has to have the members:
 *  - \c global2localIdx(unsigned,unsigned,unsigned) \c const;
 * where the first parameter is the global index and the
 * other two are the pair (local idx, rank).
 *  - \c MPI_Comm \c %communicator() \c const;  returns the communicator to use in the gather/scatter
 *  - \c local_size(); return the local vector size
 * @param global the column indices and num_cols need to be global, the row indices and num_rows local
 * @param policy the conversion object
 *
 * @return a row distributed MPI matrix. If no MPI communication is needed the collective communicator will have zero size.
 * @sa basictopology the MPI %grids defined in Level 3 can all be used as a ConversionPolicy
 * @ingroup mpi_structures
 */
template<class ConversionPolicy, class real_type>
dg::tMIHMatrix<real_type> convert( const dg::tIHMatrix<real_type>& global, const ConversionPolicy& policy)
{
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    dg::iHVec unique_global_idx;
    cusp::array1d<int, cusp::host_memory> buffer_idx;
    dg::detail::global2bufferIdx( global.column_indices, buffer_idx, unique_global_idx);
    dg::GeneralComm<dg::iHVec, thrust::host_vector<real_type>> comm( unique_global_idx, policy);
    if( !comm.isCommunicating() )
    {
        cusp::array1d<int, cusp::host_memory> local_idx(global.column_indices), pids(local_idx);
        bool success = true;
        for(unsigned i=0; i<local_idx.size(); i++)
            if( !policy.global2localIdx(global.column_indices[i], local_idx[i], pids[i]) ) success = false;
        assert( success);
        dg::tIHMatrix<real_type> local( global.num_rows, policy.local_size(), global.values.size());
        comm = dg::GeneralComm< dg::iHVec, thrust::host_vector<real_type>>();
        local.row_offsets=global.row_offsets;
        local.column_indices=local_idx;
        local.values=global.values;
        return dg::tMIHMatrix<real_type>( local, comm, dg::row_dist);
    }
    dg::tIHMatrix<real_type> local( global.num_rows, comm.size(), global.values.size());
    local.row_offsets=global.row_offsets;
    local.column_indices=buffer_idx;
    local.values=global.values;
    dg::tMIHMatrix<real_type> matrix(   local, comm, dg::row_dist);
    return matrix;
}

namespace create
{

///@addtogroup interpolation
///@{

///@copydoc dg::create::interpolation(const Grid1d&,const Grid1d&)
template<class real_type>
dg::tMIHMatrix<real_type> interpolation( const aRealMPITopology2d<real_type>& g_new, const aRealMPITopology2d<real_type>& g_old)
{
    return tMIHMatrix<real_type>( interpolation( g_new.local(), g_old.local()), GeneralComm<iHVec, thrust::host_vector<real_type>>());
}
///@copydoc interpolation(const Grid1d&,const Grid1d&)
template<class real_type>
dg::tMIHMatrix<real_type> interpolation( const aRealMPITopology3d<real_type>& g_new, const aRealMPITopology3d<real_type>& g_old)
{
    return tMIHMatrix<real_type>( interpolation( g_new.local(), g_old.local()), GeneralComm<iHVec, thrust::host_vector<real_type>>());
}

///@copydoc interpolationT(const Grid1d&,const Grid1d&)
template<class real_type>
dg::tMIHMatrix<real_type> interpolationT( const aRealMPITopology2d<real_type>& g_new, const aRealMPITopology2d<real_type>& g_old)
{
    return tMIHMatrix<real_type>( interpolationT( g_new.local(), g_old.local()), GeneralComm<iHVec, thrust::host_vector<real_type>>());
}
///@copydoc interpolationT(const Grid1d&,const Grid1d&)
template<class real_type>
dg::tMIHMatrix<real_type> interpolationT( const aRealMPITopology3d<real_type>& g_new, const aRealMPITopology3d<real_type>& g_old)
{
    return tMIHMatrix<real_type>( interpolationT( g_new.local(), g_old.local()), GeneralComm<iHVec, thrust::host_vector<real_type>>());
}

///@copydoc projection(const Grid1d&,const Grid1d&)
template<class real_type>
dg::tMIHMatrix<real_type> projection( const aRealMPITopology2d<real_type>& g_new, const aRealMPITopology2d<real_type>& g_old)
{
    return tMIHMatrix<real_type>( projection( g_new.local(), g_old.local()), GeneralComm<iHVec, thrust::host_vector<real_type>>());
}
///@copydoc projection(const Grid1d&,const Grid1d&)
template<class real_type>
dg::tMIHMatrix<real_type> projection( const aRealMPITopology3d<real_type>& g_new, const aRealMPITopology3d<real_type>& g_old)
{
    return tMIHMatrix<real_type>( projection( g_new.local(), g_old.local()), GeneralComm<iHVec, thrust::host_vector<real_type>>());
}

/**
 * @brief Create an MPI row distributed interpolation matrix
 *
 * @copydetails interpolation(const thrust::host_vector<double>&,const thrust::host_vector<double>&,const aTopology2d&,dg::bc,dg::bc)
 */
template<class real_type>
dg::tMIHMatrix<real_type> interpolation( const thrust::host_vector<real_type>& x, const thrust::host_vector<real_type>& y, const aRealMPITopology2d<real_type>& g, dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU)
{
    dg::tIHMatrix<real_type> mat = dg::create::interpolation( x,y, g.global(), bcx, bcy);
    return convert(  mat, g);
}



///@}

}//namespace create
}//namespace dg
