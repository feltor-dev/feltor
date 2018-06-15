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
template<class real_type>
MPI_Vector<thrust::host_vector<real_type> > weights( const aRealMPITopology2d<real_type>& g)
{
    thrust::host_vector<real_type> w( g.local().size());
    for( unsigned i=0; i<g.local().size(); i++)
        w[i] = g.hx()*g.hy()/4.*
                g.dlt().weights()[detail::get_i(g.n(),g.local().Nx(), i)]*
                g.dlt().weights()[detail::get_j(g.n(),g.local().Nx(), i)];
    return MPI_Vector<thrust::host_vector<real_type> >( w, g.communicator());
}
///@copydoc hide_inv_weights_doc
template<class real_type>
MPI_Vector<thrust::host_vector<real_type> > inv_weights( const aRealMPITopology2d<real_type>& g)
{
    MPI_Vector<thrust::host_vector<real_type> > v = weights( g);
    for( unsigned i=0; i<g.local().size(); i++)
        v.data()[i] = 1./v.data()[i];
    return v;
}

///@copydoc hide_weights_coo_doc
template<class real_type>
MPI_Vector<thrust::host_vector<real_type> > weights( const aRealMPITopology2d<real_type>& g, enum coo2d coo)
{
    thrust::host_vector<real_type> w( g.local().size());
    if( coo == coo2d::x) {
        for( unsigned i=0; i<g.local().size(); i++)
            w[i] = g.hx()/2.* g.dlt().weights()[i%g.n()];
    }
    else if( coo == coo2d::y) {
        for( unsigned i=0; i<g.local().size(); i++)
            w[i] = g.hy()/2.* g.dlt().weights()[(i/(g.n()*g.local().Nx()))%g.n()];
    }
    return MPI_Vector<thrust::host_vector<real_type> >( w, g.communicator());
}
///@copydoc hide_weights_doc
///@copydoc hide_code_mpi_evaluate3d
template<class real_type>
MPI_Vector<thrust::host_vector<real_type> > weights( const aRealMPITopology3d<real_type>& g)
{
    thrust::host_vector<real_type> w( g.local().size());
    for( unsigned i=0; i<g.local().size(); i++)
        w[i] = g.hx()*g.hy()*g.hz()/4.*
               g.dlt().weights()[detail::get_i(g.n(), g.local().Nx(), i)]*
               g.dlt().weights()[detail::get_j(g.n(), g.local().Nx(), i)];
    return MPI_Vector<thrust::host_vector<real_type> >( w, g.communicator());
}
///@copydoc hide_inv_weights_doc
template<class real_type>
MPI_Vector<thrust::host_vector<real_type> > inv_weights( const aRealMPITopology3d<real_type>& g)
{
    MPI_Vector<thrust::host_vector<real_type> > v = weights( g);
    for( unsigned i=0; i<g.local().size(); i++)
        v.data()[i] = 1./v.data()[i];
    return v;
}

///@copydoc hide_weights_coo_doc
template<class real_type>
MPI_Vector<thrust::host_vector<real_type> > weights( const aRealMPITopology3d<real_type>& g, enum coo3d coo)
{
    thrust::host_vector<real_type> w( g.local().size());
    if( coo == coo3d::x) {
        for( unsigned i=0; i<g.local().size(); i++)
            w[i] = g.hx()/2.* g.dlt().weights()[i%g.n()];
    }
    else if( coo == coo3d::y) {
        for( unsigned i=0; i<g.local().size(); i++)
            w[i] = g.hy()/2.* g.dlt().weights()[(i/(g.n()*g.local().Nx()))%g.n()];
    }
    else if( coo == coo3d::z) {
        for( unsigned i=0; i<g.local().size(); i++)
            w[i] = g.hz();
    }
    else if( coo == coo3d::xy) {
        for( unsigned i=0; i<g.local().size(); i++)
            w[i] = g.hx()*g.hy()/4.* g.dlt().weights()[i%g.n()]*g.dlt().weights()[(i/(g.n()*g.local().Nx()))%g.n()];
    }
    else if( coo == coo3d::yz) {
        for( unsigned i=0; i<g.local().size(); i++)
            w[i] = g.hy()*g.hz()/2.* g.dlt().weights()[(i/(g.n()*g.local().Nx()))%g.n()];
    }
    else if( coo == coo3d::xz) {
        for( unsigned i=0; i<g.local().size(); i++)
            w[i] = g.hx()*g.hz()/2.* g.dlt().weights()[i%g.n()];
    }
    return MPI_Vector<thrust::host_vector<real_type> >( w, g.communicator());
}

///@}
}//namespace create

}//namespace dg
