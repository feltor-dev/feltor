#pragma once
#include "mpi_precon.h"

namespace dg
{
namespace blas2
{
namespace detail
{
template< class Matrix, class Vector>
inline typename MatrixTraits<Matrix>::value_type doDot( const Vector& x, const Matrix& m, const Vector& y, MPIPreconTag, MPIVectorTag)
{
#ifdef DG_DEBUG
    assert( x.data().size() == y.data().size() );
    assert( x.n()*x.n() == m.data.size() );
#endif //DG_DEBUG
    typename MatrixTraits<Matrix>::value_type temp=0, sum=0;
    const unsigned n = x.n();
    for( unsigned k=0; k<x.Nz(); k++)
        for( unsigned i=n; i<(x.Ny()-1)*n; i++)
            for( unsigned j=n; j<(x.Nx()-1)*n; j++)
                    temp+=x.data()[(k*x.Ny()*n + i)*x.Nx()*n + j ]*
                          m.data[(i%n)*n+(j%n)]*
                          y.data()[(k*x.Ny()*n + i)*x.Nx()*n + j ];
    MPI_Allreduce( &temp, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    return sum;
}
template< class Matrix, class Vector>
inline typename MatrixTraits<Matrix>::value_type doDot( const Matrix& m, const Vector& x, dg::MPIPreconTag, dg::MPIVectorTag)
{
    return doDot( x, m,x, MPIPreconTag(), MPIVectorTag());
}

template< class Matrix, class Vector>
inline void doSymv(  
              typename MatrixTraits<Matrix>::value_type alpha, 
              const Matrix& m,
              const Vector& x, 
              typename MatrixTraits<Matrix>::value_type beta, 
              Vector& y, 
              MPIPreconTag,
              MPIVectorTag)
{
#ifdef DG_DEBUG
    assert( x.data().size() == y.data().size() );
#endif //DG_DEBUG
    if( alpha == 0)
    {
        if( beta == 1) 
            return;
        dg::blas1::detail::doAxpby( 0., x, beta, y, dg::MPIVectorTag());
        return;
    }
    const unsigned& n=m.data.size();
    const unsigned& size = x.data().size();
#ifdef DG_DEBUG
    assert( n >= 1);
    assert( x.data().size() == y.data().size() );
    assert( size%n ==0);
    assert( x.n()*x.n() == n);
#endif //DG_DEBUG
    for( unsigned i=0; i<x.Nz()*x.Ny(); i++)
        for( unsigned k=0; k<x.n(); k++)
            for( unsigned j=0; j<x.Nx(); j++)
                for( unsigned l=0; l<x.n(); l++)
                    y.data()[((i*x.n() + k)*x.Nx()+ j)*x.n() +l] = alpha*m.data[k*n+l]*x.data()[((i*x.n() + k)*x.Nx()+ j)*x.n() +l] + beta*y.data()[((i*x.n() + k)*x.Nx()+ j)*x.n() +l];
}

template< class Matrix, class Vector>
inline void doSymv( const Matrix& m, const Vector&x, Vector& y, MPIPreconTag, MPIVectorTag, MPIVectorTag  )
{
    doSymv( 1., m, x, 0, y);
}


} //namespace detail
} //namespace blas2
} //namespace dg
