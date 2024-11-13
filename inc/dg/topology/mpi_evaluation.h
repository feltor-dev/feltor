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
MPI_Vector<thrust::host_vector<real_type> > global2local( const thrust::host_vector<real_type>& global, const MPITopology& g)
{
    assert( global.size() == g.global().size());
    auto l = g.local();
    thrust::host_vector<real_type> temp(l.size());

    int dims[g.ndim()], periods[g.ndim()], coords[g.ndim()];
    MPI_Cart_get( g.communicator(), g.ndim(), dims, periods, coords);
    // an exercise in flattening and unflattening indices
    for( unsigned idx = 0; idx<l.size(); idx++)
    {
        // idx = i[0] + s[0]*( i[1] + s[1]*(i[2]+s[2]*(...)))
        // convert flattened index to indices
        size_t i[g.ndim()], rest = idx;
        for( unsigned d=0; d<g.ndim(); d++)
        {
            i[d] = rest%g.local().shape(d);
            rest = rest/g.local().shape(d);
        }
        size_t idxx = 0;
        // convert to
        for( unsigned d=0; d<g.ndim(); d++)
        {
            // we need to construct from inside
            unsigned dd = g.ndim()-1-d;
            // 2 for loops e.g.
            //for( unsigned pz=0; pz<dims[2]; pz++)
            //for( unsigned s=0; s<shape[2]; s++)
            idxx = (idxx*dims[dd] + coords[dd])*g.local().shape(dd)+i[dd];
        }
        temp[idx] = global[idxx];
    }
    return MPI_Vector<thrust::host_vector<real_type> >(temp, g.communicator());
}

}//namespace dg

