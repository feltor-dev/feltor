#pragma once

#include <type_traits>
#include "mpi_vector.h"
#include "tensor_traits.h"
#include "predicate.h"

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

///@cond
namespace dg {
template<class value_type>
static inline MPI_Datatype getMPIDataType();

template<class to_ContainerType, class from_ContainerType, class ...Params>
inline to_ContainerType construct( const from_ContainerType& src, Params&& ...ps);
template<class from_ContainerType, class to_ContainerType, class ...Params>
inline void assign( const from_ContainerType&, to_ContainerType&, Params&& ...ps);
namespace detail{

template< class Vector1, class Vector2, class ...Params>
Vector1 doConstruct( const Vector2& in, MPIVectorTag, MPIVectorTag, Params&& ...ps)
{
    Vector1 out;
    out.set_communicator(in.communicator(), in.communicator_mod(),
            in.communicator_mod_reduce());
    using container1 = typename std::decay_t<Vector1>::container_type;
    out.data() = dg::construct<container1>( in.data(), std::forward<Params>(ps)...);
    return out;
}
template< class Vector1, class Vector2, class ...Params>
void doAssign( const Vector1& in, Vector2& out, MPIVectorTag, MPIVectorTag, Params&& ...ps)
{
    out.set_communicator(in.communicator(), in.communicator_mod(),
            in.communicator_mod_reduce());
    dg::assign( in.data(), out.data(), std::forward<Params>(ps)...);
}

}//namespace detail
namespace blas1{

namespace detail{


template< class Vector1, class Vector2>
void do_mpi_assert( const Vector1& x, const Vector2& y, AnyVectorTag, AnyVectorTag)
{
    return;
}
template< class Vector1, class Vector2>
void do_mpi_assert( const Vector1& x, const Vector2& y, MPIVectorTag, MPIVectorTag)
{
    int compare;
    MPI_Comm_compare( x.communicator(), y.communicator(), &compare);
    assert( compare == MPI_CONGRUENT || compare == MPI_IDENT);
}
template< class Vector1, class Vector2>
void mpi_assert( const Vector1& x, const Vector2&y)
{
    do_mpi_assert( x,y, get_tensor_category<Vector1>(), get_tensor_category<Vector2>());
}

template< class Vector1, class Vector2>
std::vector<int64_t> doDot_superacc( const Vector1& x, const Vector2& y, MPIVectorTag)
{
    //find out which one is the MPIVector and determine category
    constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar, Vector1, Vector1, Vector2>::value;
#ifdef DG_DEBUG
    mpi_assert( x,y);
#endif //DG_DEBUG
    //local compuation
    std::vector<int64_t> acc = doDot_superacc(
        do_get_data(x,get_tensor_category<Vector1>()),
        do_get_data(y,get_tensor_category<Vector2>()));
    std::vector<int64_t> receive(exblas::BIN_COUNT, (int64_t)0);
    //get communicator from MPIVector
    auto comm = get_idx<vector_idx>(x,y).communicator();
    auto comm_mod = get_idx<vector_idx>(x,y).communicator_mod();
    auto comm_red = get_idx<vector_idx>(x,y).communicator_mod_reduce();
    exblas::reduce_mpi_cpu( 1, acc.data(), receive.data(), comm, comm_mod, comm_red);
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

} //namespace detail
} //namespace blas1
} //namespace dg
///@endcond
