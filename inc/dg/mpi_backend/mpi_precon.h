#pragma once
#include "mpi_config.h"


namespace dg
{
 
struct MPI_Precon
{
    std::vector<double> data;
};

template <>
struct MatrixTraits<MPI_Precon>
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
*/
MPI_Precon w1d( const Grid1d<T>& g)
{
    Precond p;
    p.data.resize( g.n());
    for( unsigned i=0; i<g.n(); i++)
        p.data[i] = g.h()/2.*g.dlt().weights()[i];
    return p;
}
/**
* @brief create Preconditioner containing 1d X-space inverse weight coefficients
*
* @param g The grid 
*
* @return Preconditioner
*/
MPI_Precon v1d( const Grid1d<T>& g)
{
    MPI_Precon p = w1d(g);
    for( unsigned i=0; i<g.n(); i++)
        p.data[i] = 1./p.data[i];
    return p;
}

///@endcond

/**
* @brief create Preconditioner containing 2d X-space weight coefficients
*
* @param g The grid 
*
* @return Preconditioner
*/
MPI_Precon w2d( const Grid2d<T>& g)
{
    MPI_Precon p;
    p.data.resize( g.n()*g.n());
    for( unsigned i=0; i<g.n(); i++)
        for( unsigned j=0; j<g.n(); j++)
            p.data[i*n+j] = g.hx()*g.hy()/4.*g.dlt().weights()[i]*g.dlt().weights()[j];
    return p;
}
/**
* @brief create Preconditioner containing 2d inverse X-space weight coefficients
*
* @param g The grid 
*
* @return Preconditioner
*/
MPI_Precon v2d( const Grid2d<T>& g)
{
    MPI_Precon v = w2d( g);
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
*/
MPI_Precon w3d( const Grid3d<T>& g)
{
    MPI_Precon p;
    p.data.resize( g.n()*g.n());
    for( unsigned i=0; i<g.n(); i++)
        for( unsigned j=0; j<g.n(); j++)
            p.data[i*n+j] = g.hz()*g.hx()*g.hy()/4.*g.dlt().weights()[i]*g.dlt().weights()[j];
    return p;
}
/**
* @brief create Preconditioner containing 3d inverse X-space weight coefficients
*
* @param g The grid 
*
* @return Preconditioner
*/
MPI_Precon v3d( const Grid3d<T>& g)
{
    MPI_Precon p = w3d( g);
    for( unsigned i=0; i<p.data.size(); i++)
        p.data[i] = 1./p.data[i];
    return p;
}

}//namespace create
}//namespace dg
