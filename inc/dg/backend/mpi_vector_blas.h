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

template< class Vector>
inline void doScal(  Vector& x,
              typename VectorTraits<Vector>::value_type alpha,
              MPIVectorTag)
{
    //local computation
    typedef typename Vector::container_type container;
    doScal( x.data(), alpha, typename VectorTraits<container>::vector_category());
}
template< class Vector>
inline void doPlus(  Vector& x,
              typename VectorTraits<Vector>::value_type alpha,
              MPIVectorTag)
{
    //local computation
    typedef typename Vector::container_type container;
    doPlus( x.data(), alpha, typename VectorTraits<container>::vector_category());
}

template< class Vector, class UnaryOp>
inline void doTransform(  const Vector& x, Vector& y,
                          UnaryOp op,
                          MPIVectorTag)
{
#ifdef DG_DEBUG
    int result;
    MPI_Comm_compare( x.communicator(), y.communicator(), &result);
    assert( result == MPI_CONGRUENT || result == MPI_IDENT);
#endif //DG_DEBUG
    typedef typename Vector::container_type container;
    doTransform( x.data(), y.data(), op, typename VectorTraits<container>::vector_category());
}

template< class Vector>
inline void doAxpby( typename VectorTraits<Vector>::value_type alpha,
              const Vector& x,
              typename VectorTraits<Vector>::value_type beta,
              Vector& y,
              MPIVectorTag)
{
    typedef typename Vector::container_type container;
    doAxpby( alpha, x.data(), beta, y.data(), typename VectorTraits<container>::vector_category());
}

template< class Vector>
inline void doAxpbypgz( typename VectorTraits<Vector>::value_type alpha,
              const Vector& x,
              typename VectorTraits<Vector>::value_type beta,
              const Vector& y,
              typename VectorTraits<Vector>::value_type gamma,
              Vector& z,
              MPIVectorTag)
{
    typedef typename Vector::container_type container;
    doAxpbypgz( alpha,x.data(),beta, y.data(), gamma, z.data(), typename VectorTraits<container>::vector_category());
}

template< class Vector>
inline void doPointwiseDot( typename VectorTraits<Vector>::value_type alpha,
        const Vector& x1, const Vector& x2,
        typename VectorTraits<Vector>::value_type beta,
        Vector& y,
        MPIVectorTag)
{
    typedef typename Vector::container_type container;
    doPointwiseDot( alpha, x1.data(), x2.data(), beta, y.data(), typename VectorTraits<container>::vector_category());
}
template< class Vector>
inline void doPointwiseDivide( typename VectorTraits<Vector>::value_type alpha,
        const Vector& x1, const Vector& x2,
        typename VectorTraits<Vector>::value_type beta,
        Vector& y,
        MPIVectorTag)
{
    typedef typename Vector::container_type container;
    doPointwiseDivide( alpha, x1.data(), x2.data(), beta, y.data(), typename VectorTraits<container>::vector_category());
}

template< class Vector>
inline void doPointwiseDot( typename VectorTraits<Vector>::value_type alpha,
        const Vector& x1, const Vector& x2, const Vector& x3,
        typename VectorTraits<Vector>::value_type beta,
        Vector& y,
        MPIVectorTag)
{
    typedef typename Vector::container_type container;
    doPointwiseDot( alpha, x1.data(), x2.data(), x3.data(), beta, y.data(), typename VectorTraits<container>::vector_category());
}

template< class Vector>
inline void doPointwiseDot( typename VectorTraits<Vector>::value_type alpha,
        const Vector& x1, const Vector& x2,
        typename VectorTraits<Vector>::value_type beta,
        const Vector& y1, const Vector& y2,
        typename VectorTraits<Vector>::value_type gamma,
        Vector& z,
        MPIVectorTag)
{
    typedef typename Vector::container_type container;
    doPointwiseDot( alpha, x1.data(), x2.data(), beta, y1.data(), y2.data(), gamma, z.data(), typename VectorTraits<container>::vector_category());

}


}//namespace detail

} //namespace blas1

} //namespace dg
///@endcond
