#pragma once

#include "weights.cuh"
#include "mpi_grid.h"



/**@file
* @brief contains MPI weights
*/
namespace dg
{
namespace create
{

///@addtogroup highlevel
///@{

///@copydoc hide_weights_doc
///@copydoc hide_code_mpi_evaluate2d
MPI_Vector<thrust::host_vector<double> > weights( const aMPITopology2d& g)
{
    thrust::host_vector<double> w( g.local().size());
    for( unsigned i=0; i<g.local().size(); i++)
        w[i] = g.hx()*g.hy()/4.*
                g.dlt().weights()[detail::get_i(g.n(),g.local().Nx(), i)]*
                g.dlt().weights()[detail::get_j(g.n(),g.local().Nx(), i)];
    return MPI_Vector<thrust::host_vector<double> >( w, g.communicator());
}
///@copydoc hide_inv_weights_doc
MPI_Vector<thrust::host_vector<double> > inv_weights( const aMPITopology2d& g)
{
    MPI_Vector<thrust::host_vector<double> > v = weights( g);
    for( unsigned i=0; i<g.local().size(); i++)
        v.data()[i] = 1./v.data()[i];
    return v;
}
///@copydoc hide_weights_doc
///@copydoc hide_code_mpi_evaluate3d
MPI_Vector<thrust::host_vector<double> > weights( const aMPITopology3d& g)
{
    thrust::host_vector<double> w( g.local().size());
    for( unsigned i=0; i<g.local().size(); i++)
        w[i] = g.hx()*g.hy()*g.hz()/4.*
               g.dlt().weights()[detail::get_i(g.n(), g.local().Nx(), i)]*
               g.dlt().weights()[detail::get_j(g.n(), g.local().Nx(), i)];
    return MPI_Vector<thrust::host_vector<double> >( w, g.communicator());
}
///@copydoc hide_inv_weights_doc
MPI_Vector<thrust::host_vector<double> > inv_weights( const aMPITopology3d& g)
{
    MPI_Vector<thrust::host_vector<double> > v = weights( g);
    for( unsigned i=0; i<g.local().size(); i++)
        v.data()[i] = 1./v.data()[i];
    return v;
}

///@}
}//namespace create

}//namespace dg
