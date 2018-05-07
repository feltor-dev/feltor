#pragma once
#include <vector>
#include "blas1_dispatch_mpi.h"
#include "thrust_matrix_blas.cuh"

namespace dg
{

///@addtogroup mat_list
///@{
template <class T>
struct MatrixTraits<MPI_Vector<T> >
{
    using value_type        = get_value_type<T>;
    using matrix_category   = MPIPreconTag;
};
template <class T>
struct MatrixTraits<const MPI_Vector<T> >
{
    using value_type        = get_value_type<T>;
    using matrix_category   = MPIPreconTag;
};
///@}

///@cond
namespace blas2
{
namespace detail
{
template< class Precon, class Vector>
inline std::vector<int64_t> doDot_superacc( const Vector& x, const Precon& P, const Vector& y, MPIPreconTag, MPIVectorTag)
{
#ifdef DG_DEBUG
    int compare;
    MPI_Comm_compare( x.communicator(), y.communicator(), &compare);
    assert( compare == MPI_CONGRUENT || compare == MPI_IDENT);
    MPI_Comm_compare( x.communicator(), P.communicator(), &compare);
    assert( compare == MPI_CONGRUENT || compare == MPI_IDENT);
#endif //DG_DEBUG
    //local compuation
    std::vector<int64_t> acc = doDot_superacc(x.data(), P.data(), y.data(), ThrustMatrixTag(), ThrustVectorTag());
    std::vector<int64_t> receive(exblas::BIN_COUNT, (int64_t)0);
    exblas::reduce_mpi_cpu( 1, acc.data(), receive.data(), x.communicator(), x.communicator_mod(), x.communicator_mod_reduce());

    return receive;
}
template< class Precon, class Vector>
inline typename MatrixTraits<Precon>::value_type doDot( const Vector& x, const Precon& P, const Vector& y, MPIPreconTag, MPIVectorTag)
{
    std::vector<int64_t> acc = doDot_superacc( x,P,y,MPIPreconTag(), MPIVectorTag());
    return exblas::cpu::Round(acc.data());
}

template< class Matrix, class Vector>
inline typename MatrixTraits<Matrix>::value_type doDot( const Matrix& m, const Vector& x, dg::MPIPreconTag, dg::MPIVectorTag)
{
    std::vector<int64_t> acc = doDot_superacc( x,m,x,MPIPreconTag(), MPIVectorTag());
    return exblas::cpu::Round(acc.data());
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
///@endcond
} //namespace dg
