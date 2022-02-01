#pragma once

#include "weights.h"
#include "mpi_grid.h"



/**@file
* @brief MPI weights
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
    thrust::host_vector<real_type> w = dg::create::weights( g.local());
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
    thrust::host_vector<real_type> w = create::weights( g.local(), coo);
    return MPI_Vector<thrust::host_vector<real_type> >( w, g.communicator());
}
///@copydoc hide_weights_doc
///@copydoc hide_code_mpi_evaluate3d
template<class real_type>
MPI_Vector<thrust::host_vector<real_type> > weights( const aRealMPITopology3d<real_type>& g)
{
    thrust::host_vector<real_type> w = weights( g.local());
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
    thrust::host_vector<real_type> w = weights( g.local(), coo);
    return MPI_Vector<thrust::host_vector<real_type> >( w, g.communicator());
}

///@}
}//namespace create

}//namespace dg
