#pragma once

#include "matrix_traits.h"
#include "weights.cuh"
#include "mpi_grid.h"



/**@file
* @brief contains MPI weights
*/
namespace dg
{
 

///@cond

template <class T>
struct MatrixTraits<MPI_Vector<T> >
{
    using value_type = get_value_type<T>;
    using matrix_category = MPIPreconTag;
};
template <class T>
struct MatrixTraits<const MPI_Vector<T> >
{
    using value_type = get_value_type<T>;
    using matrix_category = MPIPreconTag;
};
///@endcond
namespace create
{

///@addtogroup highlevel
///@{

///@copydoc hide_weights_doc
///@copydoc hide_code_mpi_evaluate2d
MPI_Vector<thrust::host_vector<double> > weights( const aMPITopology2d& g)
{
    thrust::host_vector<double> w = create::weights( g.local());
    return MPI_Vector<thrust::host_vector<double> >( w, g.communicator());
}
///@copydoc hide_inv_weights_doc
MPI_Vector<thrust::host_vector<double> > inv_weights( const aMPITopology2d& g)
{
    thrust::host_vector<double> w = create::inv_weights( g.local());
    return MPI_Vector<thrust::host_vector<double> >( w, g.communicator());
}
///@copydoc hide_weights_doc
///@copydoc hide_code_mpi_evaluate3d
MPI_Vector<thrust::host_vector<double> > weights( const aMPITopology3d& g)
{
    thrust::host_vector<double> w = create::weights( g.local());
    return MPI_Vector<thrust::host_vector<double> >( w, g.communicator());
}
///@copydoc hide_inv_weights_doc
MPI_Vector<thrust::host_vector<double> > inv_weights( const aMPITopology3d& g)
{
    thrust::host_vector<double> w = create::inv_weights( g.local());
    return MPI_Vector<thrust::host_vector<double> >( w, g.communicator());
}

///@}
}//namespace create

}//namespace dg
