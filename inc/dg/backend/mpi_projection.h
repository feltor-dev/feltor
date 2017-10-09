#pragma once

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
//buffer_idx -> (gather map/ new column indices) same size as global_idx ( can alias global_idx
//unique_global_idx -> (idx to be used in a Collective Communication object)
void global2bufferIdx( cusp::array1d<int, cusp::host_memory>& global_idx, cusp::array1d<int, cusp::host_memory>& buffer_idx, thrust::host_vector<int>& unique_global_idx)
{
    thrust::host_vector<int> index(global_idx_begin, global_idx_end);
    thrust::sequence( index.begin(), index.end());
    thrust::stable_sort_by_key( global_idx.begin(), global_idx.end(), index.begin());
    thrust::host_vector<int> ones( index.size(), 1);
    thrust::host_vector<int> unique_global( index.size()), howmany( index.size());
    thrust::pair<int*, int*> new_end;
    new_end = thrust::reduce_by_key( global_idx.begin(), global_idx.end(), ones.begin(), unique_global.begin(), howmany.begin());
    unique_global_idx.assign( unique_global.begin(), new_end.first);
    thrust::host_vector<int> gather_map;
    for( int i=0; i<unique_global_idx.size(); i++)
        for( int j=0; j<howmany[i]; j++)
            gather_map.append(i );
    assert( gather_map.size() == global_idx.size());
    buffer_idx.resize( global_idx.size());
    thrust::scatter( gather_map.begin(), gather_map.end(), index.begin(), buffer_idx.begin());
}
}//namespace detail
///@}

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
 * @brief Convert a matrix with global indices to a row distributed MPI matrix
 *
 * @param global the column indices need to be global, the row indices local
 * @param topology the topology defines how the indices are converted from global to local
 *
 * @return 
 */
dg::MIHMatrix convert( const dg::IHMatrix& global, const aMPITopology2d& topology) 
{
    dg::iHVec unique_global_idx;
    cusp::array1d<int, cusp::host_memory> buffer_idx;
    dg::detail::global2bufferIdx( global.column_indices, buffer_idx, unique_global_idx);

    dg::IHMatrix local( global.num_rows, topology.size(), global.num_values);
    local.row_offsets=global.row_offsets;
    local.column_indices=buffer_idx;
    local.values=global.values;

    dg::GeneralComm<dg::iHVec, dg::HVec> comm( unique_global_idx, topology);
    dg::MIHMatrix matrix(   local, comm, dg::row_dist);
    return matrix;
}


/**
 * @brief Create an MPI row distributed interpolation matrix 
 *
 * @copydetails interpolation(const thrust::host_vector<double>&,const thrust::host_vector<double>&,const aTopology2d&,dg::bc,dg::bc)
 */
dg::MIHMatrix interpolation( const MPI_Vector<dg::HVec>& x, const MPI_Vector<dg::HVec>& y, const aMPITopology2d& grid, dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU)
{
    return convert(  
            dg::create::interpolation( x.data(), y.data(), grid.global(), bcx, bcy), 
            grid);
}



///@}

}//namespace create
}//namespace dg
