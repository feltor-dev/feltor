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
void doTransfer( const Vector1& in, Vector2& out, MPIVectorTag, MPIVectorTag)
{
    out.communicator() = in.communicator();
    //local computation 
    typedef typename Vector1::container_type container1;
    typedef typename Vector2::container_type container2;
    doTransfer( in.data(), out.data(), typename VectorTraits<container1>::vector_category(), typename VectorTraits<container2>::vector_category());
}

template< class Vector>
typename VectorTraits<Vector>::value_type doDot( const Vector& x, const Vector& y, MPIVectorTag)
{
#ifdef DG_DEBUG
    assert( x.communicator() == y.communicator());
#endif //DG_DEBUG
    typedef typename Vector::container_type container;
    
    typename VectorTraits<Vector>::value_type sum=0;
    //local compuation
    typename VectorTraits<Vector>::value_type temp = doDot( x.data(), y.data(),typename VectorTraits<container>::vector_category());  
    //communication
    MPI_Allreduce( &temp, &sum, 1, MPI_DOUBLE, MPI_SUM, x.communicator());
    return sum;
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
    assert( x.communicator() == y.communicator());
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
inline void doAxpby( typename VectorTraits<Vector>::value_type alpha, 
              const Vector& x, 
              typename VectorTraits<Vector>::value_type beta, 
              const Vector& y, 
              Vector& z, 
              MPIVectorTag)
{
    typedef typename Vector::container_type container;
    doAxpby( alpha,x.data(),beta, y.data(), z.data(), typename VectorTraits<container>::vector_category());
}

template< class Vector>
inline void doAxpby( typename VectorTraits<Vector>::value_type alpha, 
              const Vector& x, 
              typename VectorTraits<Vector>::value_type beta, 
              const Vector& y, 
              typename VectorTraits<Vector>::value_type gamma, 
              Vector& z, 
              MPIVectorTag)
{
    typedef typename Vector::container_type container;
    doAxpby( alpha,x.data(),beta, y.data(), gamma, z.data(), typename VectorTraits<container>::vector_category());
}

template< class Vector>
inline void doPointwiseDot( const Vector& x1, const Vector& x2, Vector& y, MPIVectorTag)
{
    typedef typename Vector::container_type container;
    doPointwiseDot( x1.data(), x2.data(), y.data(), typename VectorTraits<container>::vector_category());

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
inline void doPointwiseDivide( const Vector& x1, const Vector& x2, Vector& y, MPIVectorTag)
{
    typedef typename Vector::container_type container;
    doPointwiseDivide( x1.data(), x2.data(), y.data(), typename VectorTraits<container>::vector_category());
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
