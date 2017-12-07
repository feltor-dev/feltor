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
    int compare;
    MPI_Comm_compare( x.communicator(), y.communicator(), &compare);
    assert( compare == MPI_CONGRUENT || compare == MPI_IDENT);
    MPI_Comm_compare( x.communicator(), P.communicator(), &compare);
    assert( compare == MPI_CONGRUENT || compare == MPI_IDENT);
#endif //DG_DEBUG
    //local compuation
    exblas::Superaccumulator acc_fine = doDot_dispatch(x.data(), P.data(), y.data(), ThrustMatrixTag(), ThrustVectorTag());
    acc_fine.Normalize();
    //communication (we cannot sum more than 128 accumulators at once, so we need to split)
    std::vector<int64_t> receive(39,0);
    MPI_Reduce(&(acc_fine.get_accumulator()[0]), &(receive[0]), acc_fine.get_f_words() + acc_fine.get_e_words(), MPI_LONG, MPI_SUM, 0, x.communicator_mod()); 
    int rank;
    MPI_Comm_rank( x.communicator_mod(), &rank);
    if(x.communicator_mod_reduce() != MPI_COMM_NULL)
    {
        exblas::Superaccumulator acc_reduce( receive);
        acc_reduce.Normalize();
        receive.assign(39,0);
        MPI_Reduce(&(acc_reduce.get_accumulator()[0]), &(receive[0]), acc_fine.get_f_words() + acc_fine.get_e_words(), MPI_LONG, MPI_SUM, 0, x.communicator_mod_reduce()); 
    }
    MPI_Bcast( &(receive[0]), 39, MPI_LONG, 0, x.communicator());

    return exblas::Superaccumulator(receive);
}
template< class Precon, class Vector>
inline typename MatrixTraits<Precon>::value_type doDot( const Vector& x, const Precon& P, const Vector& y, MPIPreconTag, MPIVectorTag)
{
    exblas::Superaccumulator acc_fin(doDot_dispatch( x,P,y,MPIPreconTag(),MPIVectorTag()));
    return acc_fin.Round();
}

template< class Matrix, class Vector>
inline typename MatrixTraits<Matrix>::value_type doDot( const Matrix& m, const Vector& x, dg::MPIPreconTag, dg::MPIVectorTag)
{
    exblas::Superaccumulator acc_fin(doDot_dispatch( x,m,x,MPIPreconTag(),MPIVectorTag()));
    return acc_fin.Round();
    //return doDot( x,m,x, MPIPreconTag(), MPIVectorTag());
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
