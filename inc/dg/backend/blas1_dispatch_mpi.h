#pragma once

#include "mpi_vector.h"

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

///@cond
//TODO place more asserts for debugging
namespace dg
{

namespace blas1{
namespace detail{

template< class Vector1, class Vector2>
Vector1 doTransfer( const Vector2& in, MPIVectorTag, MPIVectorTag)
{
    Vector1 out;
    out.set_communicator(in.communicator());
    typedef typename Vector1::container_type container1;
    typedef typename Vector2::container_type container2;
    out.data() = doTransfer<container1, container2>( in.data(), get_vector_category<container1>(), get_vector_category<container2>());
    return out;

}

template< class Vector>
std::vector<int64_t> doDot_superacc( const Vector& x, const Vector& y, MPIVectorTag)
{
#ifdef DG_DEBUG
    int compare;
    MPI_Comm_compare( x.communicator(), y.communicator(), &compare);
    assert( compare == MPI_CONGRUENT || compare == MPI_IDENT);
#endif //DG_DEBUG
    typedef typename Vector::container_type container;
    //local compuation
    std::vector<int64_t> acc = doDot_superacc( x.data(), y.data(),typename VectorTraits<container>::vector_category());
    std::vector<int64_t> receive(exblas::BIN_COUNT, (int64_t)0);
    exblas::reduce_mpi_cpu( 1, acc.data(), receive.data(), x.communicator(), x.communicator_mod(), x.communicator_mod_reduce());
    return receive;
}
template< class Vector>
typename VectorTraits<Vector>::value_type doDot( const Vector& x, const Vector& y, MPIVectorTag)
{
    std::vector<int64_t> acc = doDot_superacc( x,y,MPIVectorTag());
    return exblas::cpu::Round(acc.data());
}

template< class Subroutine, class container, class ...Containers>
inline void doSubroutine( MPIVectorTag, Subroutine f, container&& x, Containers&&... xs)
{
    //static check that all Containers have MPIVectorTag
    //...
#ifdef DG_DEBUG
    //is this possible?
    //int result;
    //MPI_Comm_compare( x.communicator(), y.communicator(), &result);
    //assert( result == MPI_CONGRUENT || result == MPI_IDENT);
#endif //DG_DEBUG
    using inner_container = typename container::container_type;
    doSubroutine( get_vector_category<inner_container>(), f, x.data(), xs.data()...);
}

}//namespace detail

} //namespace blas1

} //namespace dg
///@endcond
