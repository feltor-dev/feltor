#pragma once

#include "mpi_vector.h"
#include "thrust_vector_blas.cuh"

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

///@cond
//TODO place more asserts for debugging
namespace dg
{

namespace blas1{
namespace detail{

template< class Vector>
typename VectorTraits<Vector>::value_type doDot( const Vector& x, const Vector& y, MPIVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
    assert( x.communicator() == y.communicator());
#endif //DG_DEBUG
    typename VectorTraits<Vector>::value_type sum=0;
    //local compuation
    typename VectorTraits<Vector>::value_type temp = doDot( x.data(), y.data(), ThrustVectorTag());  
    //communication
    MPI_Allreduce( &temp, &sum, 1, MPI_DOUBLE, MPI_SUM, x.communicator());
    MPI_Barrier(x.communicator());

    return sum;
}
template< class Vector>
inline void doScal(  Vector& x, 
              typename VectorTraits<Vector>::value_type alpha, 
              MPIVectorTag)
{
    //local computation 
    doScal( x.data(), alpha, ThrustVectorTag());
}
template< class Vector, class UnaryOp>
inline void doTransform(  const Vector& x, Vector& y,
                          UnaryOp op,
                          MPIVectorTag)
{
    thrust::transform( x.data().begin(), x.data().end(), 
                       y.data().begin(), op);
}

template< class Vector>
inline void doAxpby( typename VectorTraits<Vector>::value_type alpha, 
              const Vector& x, 
              typename VectorTraits<Vector>::value_type beta, 
              Vector& y, 
              MPIVectorTag)
{
    doAxpby( alpha, x.data(), beta, y.data(), ThrustVectorTag());
}

template< class Vector>
inline void doAxpby( typename VectorTraits<Vector>::value_type alpha, 
              const Vector& x, 
              typename VectorTraits<Vector>::value_type beta, 
              const Vector& y, 
              Vector& z, 
              MPIVectorTag)
{
    doAxpby( alpha,x.data(),beta, y.data(), z.data(), ThrustVectorTag());
}

template< class Vector>
inline void doPointwiseDot( const Vector& x1, const Vector& x2, Vector& y, MPIVectorTag)
{
    doPointwiseDot( x1.data(), x2.data(), y.data(), ThrustVectorTag());

}

template< class Vector>
inline void doPointwiseDivide( const Vector& x1, const Vector& x2, Vector& y, MPIVectorTag)
{
    doPointwiseDivide( x1.data(), x2.data(), y.data(), ThrustVectorTag());
}
        

}//namespace detail
    
} //namespace blas1

} //namespace dg
///@endcond
