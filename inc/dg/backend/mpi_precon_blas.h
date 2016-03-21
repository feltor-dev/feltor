#pragma once
#include "mpi_vector_blas.h"
#include "mpi_precon.h"
#include "thrust_matrix_blas.cuh"

///@cond
namespace dg
{
namespace blas2
{
namespace detail
{
template< class Precon, class Vector>
inline typename MatrixTraits<Precon>::value_type doDot( const Vector& x, const Precon& P, const Vector& y, MPIPreconTag, MPIVectorTag)
{
#ifdef DG_DEBUG
    assert( x.communicator() == y.communicator());
    assert( x.communicator() == P.communicator());
#endif //DG_DEBUG
    //computation
    typename MatrixTraits<Precon>::value_type temp= doDot(x.data(), P.data(), y.data(), ThrustMatrixTag(), ThrustVectorTag());
    //communication
    typename MatrixTraits<Precon>::value_type sum=0;
    MPI_Allreduce( &temp, &sum, 1, MPI_DOUBLE, MPI_SUM, x.communicator());

    return sum;
}
template< class Matrix, class Vector>
inline typename MatrixTraits<Matrix>::value_type doDot( const Matrix& m, const Vector& x, dg::MPIPreconTag, dg::MPIVectorTag)
{
    return doDot( x, m,x, MPIPreconTag(), MPIVectorTag());
}

template< class Precon, class Vector>
inline void doSymv(  
              typename MatrixTraits<Precon>::value_type alpha, 
              const Precon& P,
              const Vector& x, 
              typename MatrixTraits<Precon>::value_type beta, 
              Vector& y, 
              MPIPreconTag,
              MPIVectorTag)
{
    doSymv( alpha, P.data(), x.data(), beta, y.data(), ThrustMatrixTag(), ThrustVectorTag());
}

template< class Matrix, class Vector>
inline void doSymv( const Matrix& m, const Vector&x, Vector& y, MPIPreconTag, MPIVectorTag, MPIVectorTag  )
{
    doSymv( 1., m, x, 0, y, MPIPreconTag(), MPIVectorTag());
}


} //namespace detail
} //namespace blas2
} //namespace dg
///@endcond
