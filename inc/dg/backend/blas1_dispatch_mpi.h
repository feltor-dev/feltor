#pragma once

#include <type_traits>
#include "exblas/mpi_accumulate.h"
#include "mpi_kron.h"
#include "mpi_vector.h"
#include "tensor_traits.h"
#include "predicate.h"

#include <cassert>

///@cond
namespace dg {
template<class value_type>
inline MPI_Datatype getMPIDataType();

template<class to_ContainerType, class from_ContainerType, class ...Params>
inline to_ContainerType construct( const from_ContainerType& src, Params&& ...ps);
template<class from_ContainerType, class to_ContainerType, class ...Params>
inline void assign( const from_ContainerType&, to_ContainerType&, Params&& ...ps);
namespace detail{

template< class Vector1, class Vector2, class ...Params>
Vector1 doConstruct( const Vector2& in, MPIVectorTag, MPIVectorTag, Params&& ...ps)
{
    Vector1 out;
    out.set_communicator(in.communicator());
    using container1 = typename std::decay_t<Vector1>::container_type;
    out.data() = dg::construct<container1>( in.data(), std::forward<Params>(ps)...);
    return out;
}
template< class Vector1, class Vector2, class ...Params>
void doAssign( const Vector1& in, Vector2& out, MPIVectorTag, MPIVectorTag, Params&& ...ps)
{
    out.set_communicator(in.communicator());
    dg::assign( in.data(), out.data(), std::forward<Params>(ps)...);
}

}//namespace detail
namespace blas1{

namespace detail{


//template< class Vector1, class Vector2>
//void do_mpi_assert( const Vector1& x, const Vector2& y, AnyVectorTag, AnyVectorTag)
//{
//    return;
//}
//template< class Vector1, class Vector2>
//void do_mpi_assert( const Vector1& x, const Vector2& y, MPIVectorTag, MPIVectorTag)
//{
//    int compare;
//    MPI_Comm_compare( x.communicator(), y.communicator(), &compare);
//    assert( compare == MPI_CONGRUENT || compare == MPI_IDENT);
//}
//template< class Vector1, class Vector2>
//void mpi_assert( const Vector1& x, const Vector2&y)
//{
//    do_mpi_assert( x,y, get_tensor_category<Vector1>(), get_tensor_category<Vector2>());
//}

template<class T, size_t N, class Functor, class ContainerType, class ...ContainerTypes>
void doDot_fpe( MPIVectorTag, int* status, std::array<T,N>& fpe, Functor f, const ContainerType& x, const ContainerTypes& ...xs)
{
    constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar, ContainerType, ContainerType, ContainerTypes...>::value;
    doDot_fpe( status, fpe, f,
        do_get_data(x, get_tensor_category<ContainerType>()),
        do_get_data(xs, get_tensor_category<ContainerTypes>())...);

    //now do the MPI reduction
    auto comm = get_idx<vector_idx>(x,xs...).communicator();
    int size;
    MPI_Comm_size( comm, &size);

    thrust::host_vector<T> reduction( size*N);
    MPI_Allgather( &fpe[0], N, getMPIDataType<T>(),
            thrust::raw_pointer_cast(reduction.data()), N, getMPIDataType<T>(),
            comm);
    //reduce received data (serial execution)
    for( unsigned k=0; k<N; k++)
        fpe[k] = T(0);
    for ( unsigned u=0; u<(unsigned)size; u++)
        for (unsigned k = 0; k < N; ++k)
            exblas::cpu::Accumulate( reduction[u*N+k], fpe, status);
    // Communicate the status to all
    MPI_Allreduce( MPI_IN_PLACE, status, 1, MPI_INT, MPI_MAX, comm);
}

template< class Vector1, class Vector2>
std::vector<int64_t> doDot_superacc( int* status, const Vector1& x, const Vector2& y, MPIVectorTag)
{
    //find out which one is the MPIVector and determine category
    constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar, Vector1, Vector1, Vector2>::value;
    //mpi_assert( x,y);
    //local compuation
    std::vector<int64_t> acc = doDot_superacc( status,
        do_get_data(x,get_tensor_category<Vector1>()),
        do_get_data(y,get_tensor_category<Vector2>()));
    std::vector<int64_t> receive(exblas::BIN_COUNT, (int64_t)0);
    //get communicator from MPIVector
    auto comm = get_idx<vector_idx>(x,y).communicator();
    MPI_Comm comm_mod, comm_red;
    dg::exblas::mpi_reduce_communicator( comm, &comm_mod, &comm_red);
    exblas::reduce_mpi_cpu( 1, acc.data(), receive.data(), comm, comm_mod, comm_red);
    // Communicate the status to all
    MPI_Allreduce( MPI_IN_PLACE, status, 1, MPI_INT, MPI_MAX, comm);
    return receive;
}


template< class Subroutine, class container, class ...Containers>
inline void doSubroutine( MPIVectorTag, Subroutine f, container&& x, Containers&&... xs)
{
    dg::blas1::subroutine( f,
        do_get_data(std::forward<container>(x), get_tensor_category<container>()),
        do_get_data(std::forward<Containers>(xs), get_tensor_category<Containers>())...);
}

template<class T, class ContainerType, class BinaryOp, class UnaryOp>
inline T doReduce( MPIVectorTag, const ContainerType& x, T zero, BinaryOp op,
        UnaryOp unary_op)
{
    T result = doReduce( get_tensor_category<decltype( x.data())>(), x.data(),
            zero, op, unary_op);
    //now do the MPI reduction
    int size;
    MPI_Comm_size( x.communicator(), &size);
    thrust::host_vector<T> reduction( size);
    MPI_Allgather( &result, 1, getMPIDataType<T>(),
            thrust::raw_pointer_cast(reduction.data()), 1, getMPIDataType<T>(),
            x.communicator());
    //reduce received data (serial execution)
    result = zero;
    for ( unsigned u=0; u<(unsigned)size; u++)
        result = op( result, reduction[u]);
    return result;
}
template< class BinarySubroutine, class Functor, class ContainerType, class ...ContainerTypes>
inline void doKronecker( MPIVectorTag, ContainerType& y, BinarySubroutine f, Functor g, const ContainerTypes&... xs)
{
    dg::blas1::kronecker( do_get_data( y, get_tensor_category<ContainerType>()), f, g,
        do_get_data(xs, get_tensor_category<ContainerTypes>())...);
}

} //namespace detail
} //namespace blas1
template<class ContainerType, class Functor, class ...ContainerTypes>
auto kronecker( Functor f, const ContainerType& x0, const ContainerTypes& ... xs);
namespace detail
{

template<class T>
inline MPI_Comm do_get_comm( const T& v, MPIVectorTag)
{
    return v.communicator();
}
template<class T>
inline MPI_Comm do_get_comm( const T& v, AnyScalarTag){
    return MPI_COMM_NULL;
}
template<class ContainerType, class Functor, class ...ContainerTypes>
auto doKronecker( MPIVectorTag, Functor f, const ContainerType& x0, const ContainerTypes& ... xs)
{
    constexpr size_t N = sizeof ...(ContainerTypes)+1;
    std::vector<MPI_Comm> comms{ do_get_comm(x0, get_tensor_category<ContainerType>()),
            do_get_comm(xs, get_tensor_category<ContainerTypes>())...};
    std::vector<MPI_Comm> non_zero_comms;

    for( unsigned u=0; u<N; u++)
        if ( comms[u] != MPI_COMM_NULL)
            non_zero_comms.push_back( comms[u]);

    auto ydata = dg::kronecker( f, do_get_data(x0, get_tensor_category<ContainerType>()), do_get_data( xs, get_tensor_category<ContainerTypes>())...);


    return MPI_Vector<decltype(ydata)>{ydata, dg::mpi_cart_kron( non_zero_comms)};
}

} //namespace detail
} //namespace dg
///@endcond
