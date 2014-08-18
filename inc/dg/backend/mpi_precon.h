#pragma once

#include "matrix_traits.h"
#include "weights.cuh"

namespace dg
{
 
struct MPI_Precon
{
    double norm;
    std::vector<double> data;
    thrust::host_vector<double> vec;
};

typedef MPI_Precon MPrecon;
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
MPI_Precon inv_weights( const MPI_Grid2d& g)
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
*/
MPI_Precon weights( const MPI_Grid3d& g)
{
    MPI_Precon p;
    p.data = g.dlt().weights();
    p.norm = g.hz()*g.hx()*g.hy()/4.;
    if( g.system() == cylindrical)
    {
        Grid1d<double> gR( g.x0(), g.x1(), g.n(), g.Nx());
        p.vec = abscissas( gR); 
    }
    return p;
}
/**
* @brief create Preconditioner containing 3d inverse X-space weight coefficients
*
* @param g The grid 
*
* @return Preconditioner
*/
MPI_Precon inv_weights( const MPI_Grid3d& g)
{
    MPI_Precon v = weights( g);
    v.norm = 1./v.norm;
    for( unsigned i=0; i<v.data.size(); i++)
        v.data[i] = 1./v.data[i];
    if( g.system() == cylindrical)
    {
        for( unsigned i=0; i<v.vec.size(); i++)
            v.vec[i] = 1./v.vec[i]; 
    }
    return v;
}

}//namespace create
}//namespace dg
