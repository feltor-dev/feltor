#pragma once

#include "dg/backend/mpi_vector.h"
#include "mpi_grid.h"
#include "evaluation.h"

/*! @file
  @brief Function discretization routines for mpi vectors
  */
namespace dg
{

/**
 * @brief Take the relevant local part of a global vector
 *
 * @param global a vector the size of the global grid
 * @param g the assumed topology
 * @return an MPI_Vector that is the distributed version of the global vector
 * @ingroup scatter
 */
template<class real_type, class MPITopology>
MPI_Vector<thrust::host_vector<real_type> > global2local(
    const thrust::host_vector<real_type>& global, const MPITopology& g)
{
    assert( global.size() == g.global().size());
    thrust::host_vector<real_type> temp(g.local().size());
    int rank;
    MPI_Comm_rank( g.communicator(), &rank);

    for( unsigned idx = 0; idx<g.local().size(); idx++)
    {
        int gIdx = 0;
        g.local2globalIdx( idx, rank, gIdx);
        temp[idx] = global[gIdx];
    }
    return MPI_Vector<thrust::host_vector<real_type> >(temp, g.communicator());
}

}//namespace dg

