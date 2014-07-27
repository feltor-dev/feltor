#pragma once
#include "mpi_config.h"


namespace dg
{
 
struct MPI_Precon
{
    double norm;
    std::vector<double> data;
};

template <>
struct MatrixTraits<MPI_Precon>
{
    typedef double value_type;
    typedef MPIPreconTag matrix_category;
};
template <>
struct MatrixTraits<const MPI_Precon>
{
    typedef double value_type;
    typedef MPIPreconTag matrix_category;
};
namespace create
{

///@cond
/**
* @brief create Preconditioner containing 1d X-space weight coefficients
*
* @param g The grid 
*
* @return Preconditioner
MPI_Precon weights( const Grid1d<double>& g)
{
    MPI_Precon p;
    p.data.resize( g.n());
    for( unsigned i=0; i<g.n(); i++)
        p.data[i] = g.h()/2.*g.dlt().weights()[i];
    return p;
}
*/
/**
* @brief create Preconditioner containing 1d X-space inverse weight coefficients
*
* @param g The grid 
*
* @return Preconditioner
MPI_Precon precond( const Grid1d<double>& g)
{
    MPI_Precon p = weights(g);
    for( unsigned i=0; i<g.n(); i++)
        p.data[i] = 1./p.data[i];
    return p;
}
*/

///@endcond

/**
* @brief create Preconditioner containing 2d X-space weight coefficients
*
* @param g The grid 
*
* @return Preconditioner
*/
MPI_Precon weights( const MPI_Grid2d& g)
{
    MPI_Precon p;
    p.data = g.dlt().weights();
    p.norm = g.hx()*g.hy()/4.;
    return p;
}
/**
* @brief create Preconditioner containing 2d inverse X-space weight coefficients
*
* @param g The grid 
*
* @return Preconditioner
*/
MPI_Precon precond( const MPI_Grid2d& g)
{
    MPI_Precon v = weights( g);
    v.norm = 1./v.norm;
    for( unsigned i=0; i<v.data.size(); i++)
        v.data[i] = 1./v.data[i];
    return v;
}
/**
* @brief create Preconditioner containing 3d X-space weight coefficients
*
* @param g The grid 
*
* @return Preconditioner
MPI_Precon weights( const Grid3d<double>& g)
{
    MPI_Precon p;
    p.data.resize( g.n()*g.n());
    for( unsigned i=0; i<g.n(); i++)
        for( unsigned j=0; j<g.n(); j++)
            p.data[i*g.n()+j] = g.hz()*g.hx()*g.hy()/4.*g.dlt().weights()[i]*g.dlt().weights()[j];
    return p;
}
*/
/**
* @brief create Preconditioner containing 3d inverse X-space weight coefficients
*
* @param g The grid 
*
* @return Preconditioner
MPI_Precon precond( const Grid3d<double>& g)
{
    MPI_Precon p = weights( g);
    for( unsigned i=0; i<p.data.size(); i++)
        p.data[i] = 1./p.data[i];
    return p;
}
*/

}//namespace create
}//namespace dg
