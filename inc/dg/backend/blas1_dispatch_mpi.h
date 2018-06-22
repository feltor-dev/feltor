#pragma once

#include <type_traits>
#include "mpi_vector.h"
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

template<class T>
inline auto get_data( T&& v, AnyVectorTag)-> decltype(v.data()){
    return v.data();
}
template<class T>
inline T get_data( T&& v, AnyScalarTag){
    return v;
}


template< class Vector1, class Vector2>
void do_assert( const Vector1& x, const Vector2& y, MPIVectorTag, AnyScalarTag)
{
    return;
}
template< class Vector1, class Vector2>
void do_assert( const Vector1& x, const Vector2& y, MPIVectorTag, MPIVectorTag)
{
    int compare;
    MPI_Comm_compare( x.communicator(), y.communicator(), &compare);
    assert( compare == MPI_CONGRUENT || compare == MPI_IDENT);
}

template< class Vector1, class Vector2>
std::vector<int64_t> doDot_superacc( const Vector1& x, const Vector2& y, MPIVectorTag)
{
#ifdef DG_DEBUG
    do_assert( x,y, get_tensor_category<Vector1>(), get_tensor_category<Vector2>());
#endif //DG_DEBUG
    using inner_container1 = typename std::decay<Vector1>::type::container_type;
    using inner_container2 = typename std::decay<Vector2>::type::container_type;
    //local compuation
    std::vector<int64_t> acc = doDot_superacc( x.data(), get_data(y), get_tensor_category<inner_container1>() );
    std::vector<int64_t> receive(exblas::BIN_COUNT, (int64_t)0);
    exblas::reduce_mpi_cpu( 1, acc.data(), receive.data(), x.communicator(), x.communicator_mod(), x.communicator_mod_reduce());
    return receive;
}

template< class Vector1, class Vector2>
get_value_type<Vector1> doDot( const Vector1& x, const Vector2& y, MPIVectorTag)
{
    static_assert( std::is_base_of<MPIVectorTag,
        get_tensor_category<Vector2>>::value || is_scalar<Vector2>::value,
        "All data layouts must derive from the same vector category (MPIVectorTag in this case)!");
    std::vector<int64_t> acc = doDot_superacc( x,y,MPIVectorTag());
    return exblas::cpu::Round(acc.data());
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
