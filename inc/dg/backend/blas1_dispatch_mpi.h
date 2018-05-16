#pragma once

#include "mpi_vector.h"

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
    out.data() = doTransfer<container1, container2>( in.data(), get_data_layout<container1>(), get_data_layout<container2>());
    return out;

}

template< class Vector, Vector2>
std::vector<int64_t> doDot_superacc( const Vector& x, const Vector2& y, MPIVectorTag)
{
#ifdef DG_DEBUG
    int compare;
    MPI_Comm_compare( x.communicator(), y.communicator(), &compare);
    assert( compare == MPI_CONGRUENT || compare == MPI_IDENT);
#endif //DG_DEBUG
    using inner_container = typename std::decay<Vector>::type::container_type;
    using inner_container2 = typename std::decay<Vector2>::type::container_type;
    //local compuation
    std::vector<int64_t> acc = doDot_superacc( x.data(), y.data(), get_data_layout<inner_container2>() );
    std::vector<int64_t> receive(exblas::BIN_COUNT, (int64_t)0);
    exblas::reduce_mpi_cpu( 1, acc.data(), receive.data(), x.communicator(), x.communicator_mod(), x.communicator_mod_reduce());
    return receive;
}

template< class Vector1, class Vector2>
get_value_type<Vector1> doDot( const Vector1& x, const Vector2& y, MPIVectorTag)
{
    static_assert( all_true<std::is_base_of<MPIVectorTag,
        get_data_layout<Vector2>>::value>::value,
        "All data layouts must derive from the same vector category (MPIVectorTag in this case)!");
    std::vector<int64_t> acc = doDot_superacc( x,y,MPIVectorTag());
    return exblas::cpu::Round(acc.data());
}

template< class Subroutine, class container, class ...Containers>
inline void doSubroutine( MPIVectorTag, Subroutine f, container&& x, Containers&&... xs)
{
    static_assert( all_true<std::is_base_of<MPIVectorTag,
        get_data_layout<Containers>>::value...>::value,
        "All data layouts must derive from the same vector category (MPIVectorTag in this case)!");
#ifdef DG_DEBUG
    //is this possible?
    //int result;
    //MPI_Comm_compare( x.communicator(), y.communicator(), &result);
    //assert( result == MPI_CONGRUENT || result == MPI_IDENT);
#endif //DG_DEBUG
    using inner_container = typename std::decay<container>::type::container_type;
    doSubroutine( get_data_layout<inner_container>(), f, x.data(), xs.data()...);
}

} //namespace detail
} //namespace blas1
} //namespace dg
///@endcond
