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
    assert( x.stride() == m.data.size() );
#endif //DG_DEBUG
    typename MatrixTraits<Matrix>::value_type temp=0, sum=0;
    for( unsigned k=0; k<x.Nz(); k++)
        for( unsigned i=1; i<x.Ny()-1; i++)
            for( unsigned j=1; j<x.Nx()-1; j++)
                for( unsigned l=0; l<x.stride(); l++)
                    temp+=x.data()[((k*x.Ny() + i)*x.Nx() + j)*x.stride() + l]*m.data[l]*
                          y.data()[((k*x.Ny() + i)*x.Nx() + j)*x.stride() + l];
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
    const unsigned& stride=m.data.size();
    const unsigned& size = x.data().size();
#ifdef DG_DEBUG
    assert( stride >= 1);
    assert( x.data().size() == y.data().size() );
    assert( size%stride ==0);
    assert( x.stride() == stride);
#endif //DG_DEBUG
    for( unsigned i=0; i<size; i++)
        y.data()[i] = alpha*m.data[i%stride]*x.data()[i] + beta*y.data()[i];
}

template< class Matrix, class Vector>
inline void doSymv( const Matrix& m, const Vector&x, Vector& y, MPIPreconTag, MPIVectorTag, MPIVectorTag  )
{
    const unsigned& stride=m.data.size();
    const unsigned& size = x.data().size();
#ifdef DG_DEBUG
    assert( stride >= 1);
    assert( x.data().size() == y.data().size() );
    assert( size%stride ==0);
    assert( x.stride() == stride);
#endif //DG_DEBUG
    for( unsigned i=0; i<size; i++)
        y.data()[i] = m.data[i%stride]*x.data()[i];
}


} //namespace detail
} //namespace blas2
} //namespace dg
