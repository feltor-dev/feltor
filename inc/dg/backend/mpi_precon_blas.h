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
inline exblas::Superaccumulator doDot_dispatch( const Vector& x, const Precon& P, const Vector& y, MPIPreconTag, MPIVectorTag)
{
#ifdef DG_DEBUG
    int result;
    MPI_Comm_compare( x.communicator(), y.communicator(), &result);
    assert( result == MPI_CONGRUENT || result == MPI_IDENT);
    MPI_Comm_compare( x.communicator(), P.communicator(), &result);
    assert( result == MPI_CONGRUENT || result == MPI_IDENT);
#endif //DG_DEBUG
    typedef typename Vector::container_type container;
    //local compuation
    Superacc acc = doDot_dispatch(x.data(), P.data(), y.data(), ThrustMatrixTag(), ThrustVectorTag());
    acc.Normalize();
    //communication
    std::vector<int64_t> result(acc.get_f_words() + acc.get_e_words(), 0);
    MPI_Allreduce(&(acc.get_accumulator()[0]), &(result[0]), acc.get_f_words() + acc.get_e_words(), MPI_LONG, MPI_SUM, 0, x.communicator()); 
    exblas::Superaccumulator acc_fin(result);
    return acc_fin;
}
template< class Precon, class Vector>
inline typename MatrixTraits<Precon>::value_type doDot( const Vector& x, const Precon& P, const Vector& y, MPIPreconTag, MPIVectorTag)
{
    exblas::Superaccumulator acc_fin(doDot_dispatch( x,P,y,MPIPreconTag(),MPIVectorTag());
    double sum = acc_fin.Round();
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
