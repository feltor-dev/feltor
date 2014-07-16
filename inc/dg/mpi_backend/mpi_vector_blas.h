#pragma once

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

namespace dg
{

namespace blas1{
namespace detail{

template< class Vector>
typename Vector::value_type doDot( const Vector& x, const Vector& y, MPIVectorTag)
{
#ifdef DG_DEBUG
    assert( x.data.size() == y.data.size() );
    assert( x.stride == y.stride);
    assert( x.Nx == y.Nx);
    assert( x.Ny == y.Ny);
    assert( x.Nz == y.Nz);
#endif //DG_DEBUG
    double temp=0, sum;
    for( unsigned k=0; k<x.Nz; k++)
        for( unsigned i=1; i<x.Ny-1; i++)
            for( unsigned j=1; j<x.Nx-1; j++)
                for( unsigned l=0; l<x.stride; l++)
                    temp+=x.data[(k*x.Nx*x.Ny + i*x.Nx + j)*x.stride + l]*
                          y.data[(k*x.Nx*x.Ny + i*x.Nx + j)*x.stride + l];
    MPI_Allreduce( &temp, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return sum;
}
template< class Vector>
inline void doScal(  Vector& x, 
              typename Vector::value_type alpha, 
              MPIVectorTag)
{
    for( unsigned i=0; i<x.data.size(); i++)
        x.data[i]*=alpha;
}

template< class Vector>
inline void doAxpby( typename Vector::value_type alpha, 
              const Vector& x, 
              typename Vector::value_type beta, 
              Vector& y, 
              MPIVectorTag)
{
#ifdef DG_DEBUG
    assert( x.data.size() == y.data.size() );
#endif //DG_DEBUG
    if( alpha == 0)
    {
        if( beta == 1) 
            return;
        for( unsigned i=0; i<y.data.size(); i++)
            y.data[i]*=beta;
        return;
    }
    for( unsigned i=0; i<x.data.size(); i++)
        y.data[i] = alpha*x.data[i]+beta*y.data[i];
}

template< class Vector>
inline void doAxpby( typename Vector::value_type alpha, 
              const Vector& x, 
              typename Vector::value_type beta, 
              const Vector& y, 
              Vector& z, 
              MPIVectorTag)
{
#ifdef DG_DEBUG
    assert( x.data.size() == y.data.size() );
    assert( x.data.size() == z.data.size() );
#endif //DG_DEBUG
    if( alpha == 0)
    {
        for( unsigned i=0; i<y.data.size(); i++)
            z.data[i] =beta*y.data[i];
        return;
    }
    if( beta == 0)
    {
        for( unsigned i=0; i<x.data.size(); i++)
            z.data[i] =alpha*x.data[i];
        return;
    }
    for( unsigned i=0; i<x.data.size(); i++)
        z.data[i] = alpha*x.data[i]+beta*y.data[i];
}

template< class Vector>
inline void doPointwiseDot( const Vector& x1, const Vector& x2, Vector& y, MPIVectorTag)
{
#ifdef DG_DEBUG
    assert( x1.data.size() == x2.data.size() );
    assert( x1.data.size() == y.data.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x1.data.size(); i++)
        y.data[i]=x1.data[i]*x2.data[i];

}

template< class Vector>
inline void doPointwiseDivide( const Vector& x1, const Vector& x2, Vector& y, MPIVectorTag)
{
#ifdef DG_DEBUG
    assert( x1.size() == x2.size() );
    assert( x1.size() == y.size() );
#endif //DG_DEBUG
    for( unsigned i=0; i<x1.data.size(); i++)
        y.data[i]=x1.data[i]/x2.data[i];
}
        

}//namespace detail
    
} //namespace blas1

} //namespace dg
