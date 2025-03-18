#pragma once
#include <vector>
#include "exblas/mpi_accumulate.h"
#include "mpi_matrix.h"
#include "blas1_dispatch_mpi.h"
#include "blas2_dispatch_shared.h"

///@cond
namespace dg
{
namespace blas2
{
template< class Stencil, class ContainerType, class ...ContainerTypes>
inline void parallel_for( Stencil f, unsigned N, ContainerType&& x, ContainerTypes&&... xs);
namespace detail
{

template< class Vector1, class Matrix, class Vector2 >
inline std::vector<int64_t> doDot_superacc( int* status, const Vector1& x, const Matrix& m, const Vector2& y, AnyScalarTag, MPIVectorTag)
{
    //find out which one is the MPIVector and determine category
    constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar, Vector1, Vector1, Vector2>::value;
    //dg::blas1::detail::mpi_assert( x,y);
    //local computation
    std::vector<int64_t> acc = doDot_superacc( status,
        do_get_data(x, get_tensor_category<Vector1>()), m,
        do_get_data(y, get_tensor_category<Vector2>()));
    std::vector<int64_t> receive(exblas::BIN_COUNT, (int64_t)0);
    //get communicator from MPIVector
    auto comm = get_idx<vector_idx>(x,y).communicator();
    MPI_Comm comm_mod, comm_red;
    dg::exblas::mpi_reduce_communicator( comm, &comm_mod, &comm_red);
    exblas::reduce_mpi_cpu( 1, acc.data(), receive.data(), comm, comm_mod, comm_red);
    MPI_Allreduce( MPI_IN_PLACE, status, 1, MPI_INT, MPI_MAX, comm);
    return receive;
}
template< class Vector1, class Matrix, class Vector2 >
inline std::vector<int64_t> doDot_superacc( int* status, const Vector1& x, const Matrix& m, const Vector2& y, MPIVectorTag, MPIVectorTag)
{
    //dg::blas1::detail::mpi_assert( m,x);
    //dg::blas1::detail::mpi_assert( m,y);
    //local computation
    std::vector<int64_t> acc = doDot_superacc( status,
        do_get_data(x, get_tensor_category<Vector1>()),
        m.data(),
        do_get_data(y, get_tensor_category<Vector2>()));
    std::vector<int64_t> receive(exblas::BIN_COUNT, (int64_t)0);
    MPI_Comm comm_mod, comm_red;
    dg::exblas::mpi_reduce_communicator( m.communicator(), &comm_mod, &comm_red);
    exblas::reduce_mpi_cpu( 1, acc.data(), receive.data(), m.communicator(),
        comm_mod, comm_red);
    MPI_Allreduce( MPI_IN_PLACE, status, 1, MPI_INT, MPI_MAX, m.communicator());

    return receive;
}
template< class Vector1, class Matrix, class Vector2>
inline std::vector<int64_t> doDot_superacc( int* status, const Vector1& x, const Matrix& m, const Vector2& y, MPIVectorTag, RecursiveVectorTag)
{
    //find out which one is the RecursiveVector and determine category
    constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar, Vector1, Vector1, Vector2>::value;
    auto size = get_idx<vector_idx>(x,y).size();
    std::vector<std::vector<int64_t>> acc( size);
    for( unsigned i=0; i<size; i++)
        acc[i] = doDot_superacc( status,
            do_get_vector_element(x,i,get_tensor_category<Vector1>()), m,
            do_get_vector_element(y,i,get_tensor_category<Vector2>()));
    for( unsigned i=1; i<size; i++)
    {
        int imin = exblas::IMIN, imax = exblas::IMAX;
        exblas::cpu::Normalize( &(acc[0][0]), imin, imax);
        imin = exblas::IMIN, imax = exblas::IMAX;
        exblas::cpu::Normalize( &(acc[i][0]), imin, imax);
        for( int k=exblas::IMIN; k<=exblas::IMAX; k++)
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
    dg::blas2::symv( m.data(), do_get_data(x, get_tensor_category<Vector1>()), do_get_data(y, get_tensor_category<Vector2>()));
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
    dg::blas2::symv( alpha, m.data(), do_get_data(x,get_tensor_category<Vector1>()), beta, do_get_data(y, get_tensor_category<Vector2>()));
}
template< class Matrix, class Vector1, class Vector2>
inline void doSymv( Matrix&& m, const Vector1& x, Vector2& y, MPIVectorTag, RecursiveVectorTag )
{
    for( unsigned i=0; i<y.size(); i++)
        dg::blas2::symv( std::forward<Matrix>(m), do_get_vector_element(x,i,get_tensor_category<Vector1>()), do_get_vector_element(y,i,get_tensor_category<Vector2>()));
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
        dg::blas2::symv( alpha, std::forward<Matrix>(m), do_get_vector_element(x,i,get_tensor_category<Vector1>()), beta, do_get_vector_element(y,i,get_tensor_category<Vector2>()));
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
        dg::blas2::symv( std::forward<Matrix>(m), do_get_vector_element(x,i,get_tensor_category<Vector1>()), do_get_vector_element(y,i,get_tensor_category<Vector2>()));
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
template< class Matrix, class Vector1, class Vector2>
inline void doSymv( Matrix&& m, const Vector1& x, Vector2& y, MPIMatrixTag, StdMapTag )
{
    for( auto el : y)
        dg::blas2::symv( std::forward<Matrix>(m), do_get_vector_element(x,el.first,get_tensor_category<Vector1>()), do_get_vector_element(y,el.first,get_tensor_category<Vector2>()));
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv( get_value_type<Vector1> alpha,
                Matrix&& m,
                const Vector1& x,
                get_value_type<Vector1> beta,
                Vector2& y,
                MPIMatrixTag,
                StdMapTag
                )
{
    for( auto el : y)
        dg::blas2::symv( alpha, std::forward<Matrix>(m), do_get_vector_element(x,el.first,get_tensor_category<Vector1>()), beta, do_get_vector_element(y,el.first,get_tensor_category<Vector2>()));
}

template<class Functor, class Matrix, class Vector1, class Vector2>
inline void doStencil(
                    Functor f,
                    Matrix&& m,
                    const Vector1&x,
                    Vector2& y,
                    MPIMatrixTag,
                    MPIVectorTag  )
{
    m.stencil( f, x,y);
}


} //namespace detail
} //namespace blas2
} //namespace dg
///@endcond
