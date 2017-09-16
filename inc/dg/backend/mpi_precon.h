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
    typedef double value_type;
    typedef MPIPreconTag matrix_category;
};
template <class T>
struct MatrixTraits<const MPI_Vector<T> >
{
    typedef double value_type;
    typedef MPIPreconTag matrix_category;
};
///@endcond
namespace create
{

///@addtogroup highlevel
///@{

/**
* @brief create Preconditioner containing 2d X-space weight coefficients
*
* @param g The grid 
*
* @return Preconditioner
*/
MPI_Vector<thrust::host_vector<double> > weights( const aMPITopology2d& g)
{
    thrust::host_vector<double> w = create::weights( g.local());
    return MPI_Vector<thrust::host_vector<double> >( w, g.communicator());
}
/**
* @brief create Preconditioner containing 2d inverse X-space weight coefficients
*
* @param g The grid 
*
* @return Preconditioner
*/
MPI_Vector<thrust::host_vector<double> > inv_weights( const aMPITopology2d& g)
{
    thrust::host_vector<double> w = create::inv_weights( g.local());
    return MPI_Vector<thrust::host_vector<double> >( w, g.communicator());
}
/**
* @brief create Preconditioner containing 3d X-space weight coefficients
*
* @param g The grid 
*
* @return Preconditioner
*/
MPI_Vector<thrust::host_vector<double> > weights( const aMPITopology3d& g)
{
    thrust::host_vector<double> w = create::weights( g.local());
    return MPI_Vector<thrust::host_vector<double> >( w, g.communicator());
}
/**
* @brief create Preconditioner containing 3d inverse X-space weight coefficients
*
* @param g The grid 
*
* @return Preconditioner
*/
MPI_Vector<thrust::host_vector<double> > inv_weights( const aMPITopology3d& g)
{
    thrust::host_vector<double> w = create::inv_weights( g.local());
    return MPI_Vector<thrust::host_vector<double> >( w, g.communicator());
}

///@}
}//namespace create

}//namespace dg
