#pragma once
#include <vector>
#include "mpi_matrix.h"
#include "blas1_dispatch_mpi.h"
#include "blas2_dispatch_shared.h"

///@cond
namespace dg
{
namespace blas2
{
//forward declare blas2 symv functions
template< class MatrixType, class ContainerType1, class ContainerType2>
void symv( MatrixType&& M,
                  const ContainerType1& x,
                  ContainerType2& y);
template< class MatrixType, class ContainerType1, class ContainerType2>
void symv( get_value_type<ContainerType1> alpha,
                  MatrixType&& M,
                  const ContainerType1& x,
                  get_value_type<ContainerType1> beta,
                  ContainerType2& y);
namespace detail
{

template< class Vector1, class Matrix, class Vector2 >
inline std::vector<int64_t> doDot_superacc( const Vector1& x, const Matrix& m, const Vector2& y, AnyScalarTag, MPIVectorTag)
{
    //find out which one is the MPIVector and determine category
    constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar, Vector1, Vector1, Vector2>::value;
#ifdef DG_DEBUG
    dg::blas1::detail::mpi_assert( x,y);
#endif //DG_DEBUG
    //local computation
    std::vector<int64_t> acc = doDot_superacc( get_data(x), m, get_data(y));
    std::vector<int64_t> receive(exblas::BIN_COUNT, (int64_t)0);
    //get communicator from MPIVector
    auto comm = std::get<vector_idx>(std::forward_as_tuple(x,y)).communicator();
    auto comm_mod = std::get<vector_idx>(std::forward_as_tuple(x,y)).communicator_mod();
    auto comm_red = std::get<vector_idx>(std::forward_as_tuple(x,y)).communicator_mod_reduce();
    exblas::reduce_mpi_cpu( 1, acc.data(), receive.data(), comm, comm_mod, comm_red);
    return receive;
}
template< class Vector1, class Matrix, class Vector2 >
inline std::vector<int64_t> doDot_superacc( const Vector1& x, const Matrix& m, const Vector2& y, MPIVectorTag, MPIVectorTag)
{
#ifdef DG_DEBUG
    dg::blas1::detail::mpi_assert( m,x);
    dg::blas1::detail::mpi_assert( m,y);
#endif //DG_DEBUG
    //local computation
    std::vector<int64_t> acc = doDot_superacc( get_data(x), m.data(), get_data(y));
    std::vector<int64_t> receive(exblas::BIN_COUNT, (int64_t)0);
    exblas::reduce_mpi_cpu( 1, acc.data(), receive.data(), m.communicator(), m.communicator_mod(), m.communicator_mod_reduce());

    return receive;
}
template< class Vector1, class Matrix, class Vector2>
inline std::vector<int64_t> doDot_superacc( const Vector1& x, const Matrix& m, const Vector2& y, MPIVectorTag, RecursiveVectorTag)
{
    //find out which one is the RecursiveVector and determine category
    constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar, Vector1, Vector1, Vector2>::value;
    auto size = std::get<vector_idx>(std::forward_as_tuple(x,y)).size();
    std::vector<std::vector<int64_t>> acc( size);
    for( unsigned i=0; i<size; i++)
        acc[i] = doDot_superacc( get_vector_element(x,i), m, get_vector_element(y,i));
    for( unsigned i=1; i<size; i++)
    {
        int imin = exblas::IMIN, imax = exblas::IMAX;
        exblas::cpu::Normalize( &(acc[0][0]), imin, imax);
        imin = exblas::IMIN, imax = exblas::IMAX;
        exblas::cpu::Normalize( &(acc[i][0]), imin, imax);
        for( int k=exblas::IMIN; k<exblas::IMAX; k++)
            acc[0][k] += acc[i][k];
    }
    return acc[0];
}

template< class Matrix1, class Matrix2>
inline void doTransfer( const Matrix1& m1, Matrix2& m2, AnyMatrixTag, MPIMatrixTag)
{
    Matrix2 m(m1);
    m2 = m;
}

//Matrix = MPI Vector
template< class Matrix, class Vector1, class Vector2>
inline void doSymv( Matrix&& m, const Vector1& x, Vector2& y, MPIVectorTag, MPIVectorTag )
{
    dg::blas2::symv( m.data(), get_data(x), get_data(y));
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv( get_value_type<Vector1> alpha,
                Matrix&& m,
                const Vector1& x,
                get_value_type<Vector1> beta,
                Vector2& y,
                MPIVectorTag,
                MPIVectorTag
                )
{
    dg::blas2::symv( alpha, m.data(), get_data(x), beta, get_data(y));
}
template< class Matrix, class Vector1, class Vector2>
inline void doSymv( Matrix&& m, const Vector1& x, Vector2& y, MPIVectorTag, RecursiveVectorTag )
{
    for( unsigned i=0; i<y.size(); i++)
        dg::blas2::symv( std::forward<Matrix>(m), get_vector_element(x,i), get_vector_element(y,i));
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv( get_value_type<Vector1> alpha,
                Matrix&& m,
                const Vector1& x,
                get_value_type<Vector1> beta,
                Vector2& y,
                MPIVectorTag,
                RecursiveVectorTag
                )
{
    for( unsigned i=0; i<y.size(); i++)
        dg::blas2::symv( alpha, std::forward<Matrix>(m), get_vector_element(x,i), beta, get_vector_element(y,i));
}
//Matrix is an MPI matrix
template< class Matrix, class Vector1, class Vector2>
inline void doSymv( Matrix&& m, const Vector1& x, Vector2& y, MPIMatrixTag, MPIVectorTag )
{
    m.symv( x, y);
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv( get_value_type<Vector1> alpha,
                Matrix&& m,
                const Vector1& x,
                get_value_type<Vector1> beta,
                Vector2& y,
                MPIMatrixTag,
                MPIVectorTag
                )
{
    m.symv( alpha, x, beta, y);
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv( Matrix&& m, const Vector1& x, Vector2& y, MPIMatrixTag, RecursiveVectorTag )
{
    for( unsigned i=0; i<y.size(); i++)
        dg::blas2::symv( std::forward<Matrix>(m), get_vector_element(x,i), get_vector_element(y,i));
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv( get_value_type<Vector1> alpha,
                Matrix&& m,
                const Vector1& x,
                get_value_type<Vector1> beta,
                Vector2& y,
                MPIMatrixTag,
                RecursiveVectorTag
                )
{
    for( unsigned i=0; i<y.size(); i++)
        dg::blas2::symv( alpha, std::forward<Matrix>(m), x[i], beta, y[i]);
}


} //namespace detail
} //namespace blas2
} //namespace dg
///@endcond
