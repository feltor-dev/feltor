#pragma once
#include <vector>
#include "blas1_dispatch_mpi.h"
#include "thrust_matrix_blas.cuh"

namespace dg
{

///@cond
namespace blas2
{
namespace detail
{
template< class Vector1, class Precon, class Vector2 >
inline std::vector<int64_t> doDot_superacc( const Vector1& x, const Precon& P, const Vector2& y, MPIVectorTag, MPIVectorTag, MPIVectorTag)
{
#ifdef DG_DEBUG
    int compare;
    MPI_Comm_compare( x.communicator(), y.communicator(), &compare);
    assert( compare == MPI_CONGRUENT || compare == MPI_IDENT);
    MPI_Comm_compare( x.communicator(), P.communicator(), &compare);
    assert( compare == MPI_CONGRUENT || compare == MPI_IDENT);
#endif //DG_DEBUG
    using inner_container1 = typename std::decay<Vector1>::type::container_type;
    using inner_container2 = typename std::decay<Precon>::type::container_type;
    using inner_container3 = typename std::decay<Vector2>::type::container_type;
    //local computation
    std::vector<int64_t> acc = doDot_superacc(x.data(), P.data(), y.data(),
        get_data_layout<inner_container1>(),
        get_data_layout<inner_container2>(),
        get_data_layout<inner_container3>());
    std::vector<int64_t> receive(exblas::BIN_COUNT, (int64_t)0);
    exblas::reduce_mpi_cpu( 1, acc.data(), receive.data(), x.communicator(), x.communicator_mod(), x.communicator_mod_reduce());

    return receive;
}
template< class Vector1, class Precon, class Vector2 >
inline typename MatrixTraits<Precon>::value_type doDot( const Vector1& x, const Precon& P, const Vector2& y, MPIVectorTag, MPIVectorTag, MPIVectorTag)
{
    std::vector<int64_t> acc = doDot_superacc( x,P,y,
        MPIVectorTag(), MPIVectorTag(), MPIVectorTag());
    return exblas::cpu::Round(acc.data());
}

template< class Matrix, class Vector>
inline typename MatrixTraits<Matrix>::value_type doDot( const Matrix& m, const Vector& x, dg::MPIVectorTag, dg::MPIVectorTag)
{
    std::vector<int64_t> acc = doDot_superacc( x,m,x,MPIVectorTag(), MPIVectorTag());
    return exblas::cpu::Round(acc.data());
}

template< class Matrix1, class Matrix2>
inline void doTransfer( const Matrix1& m1, Matrix2& m2, MPIMatrixTag, MPIMatrixTag)
{
    Matrix2 m(m1);
    m2 = m;
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv( Matrix& m, Vector1& x, Vector2& y, MPIMatrixTag, MPIVectorTag, MPIVectorTag )
{
    m.symv( x, y);
}

template< class Matrix, class Vector>
inline void doSymv( typename MatrixTraits<Matrix>::value_type alpha, const Matrix& m, const Vector& x, typename MatrixTraits<Matrix>::value_type beta, Vector& y, MPIMatrixTag, MPIVectorTag )
{
    m.symv( alpha, x, beta, y);
}

template< class Matrix, class Vector>
inline void doSymv( Matrix& m, Vector& x, Vector& y, CuspMatrixTag, MPIVectorTag, MPIVectorTag )
{
    typedef typename Vector::container_type container;
    doSymv(m,x.data(),y.data(),CuspMatrixTag(),typename TypeTraits<container>::data_layout(),
                                             typename TypeTraits<container>::data_layout());
}


} //namespace detail
} //namespace blas2
///@endcond
} //namespace dg
