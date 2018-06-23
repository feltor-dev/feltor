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
namespace blas1{
namespace detail{

template< class Vector1, class Vector2>
Vector1 doTransfer( const Vector2& in, MPIVectorTag, MPIVectorTag)
{
    Vector1 out;
    out.set_communicator(in.communicator());
    using container1 = typename std::decay<Vector1>::type::container_type;
    using container2 = typename std::decay<Vector2>::type::container_type;
    out.data() = doTransfer<container1, container2>( in.data(), get_tensor_category<container1>(), get_tensor_category<container2>());
    return out;

}

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
std::vector<int64_t> doDot_superacc( const Vector1& x, const Vector2& y, MPIVectorTag)
{
    //find out which one is the MPIVector and determine category
    constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar, Vector1, Vector1, Vector2>::value;
#ifdef DG_DEBUG
    do_mpi_assert( x,y, get_tensor_category<Vector1>(), get_tensor_category<Vector2>());
#endif //DG_DEBUG
    //local compuation
    std::vector<int64_t> acc = doDot_superacc( get_data(x, get_tensor_category<Vector1>()), get_data(y, get_tensor_category<Vector2>()));
    std::vector<int64_t> receive(exblas::BIN_COUNT, (int64_t)0);
    //get communicator from MPIVector
    auto comm = std::get<vector_idx>(std::forward_as_tuple(x,y)).communicator();
    auto comm_mod = std::get<vector_idx>(std::forward_as_tuple(x,y)).communicator_mod();
    auto comm_red = std::get<vector_idx>(std::forward_as_tuple(x,y)).communicator_mod_reduce();
    exblas::reduce_mpi_cpu( 1, acc.data(), receive.data(), comm, comm_mod, comm_red);
    return receive;
}


template< class Subroutine, class container, class ...Containers>
inline void doSubroutine( MPIVectorTag, Subroutine f, container&& x, Containers&&... xs)
{
    dg::blas1::subroutine( f,
        get_data(std::forward<container>(x), get_tensor_category<container>()),
        get_data(std::forward<Containers>(xs), get_tensor_category<Containers>())...);
}

} //namespace detail
} //namespace blas1
} //namespace dg
///@endcond
