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
typename VectorTraits<Vector>::value_type doDot( const Vector& x, const Vector& y, MPIVectorTag)
{
#ifdef DG_DEBUG
    int compare;
    MPI_Comm_compare( x.communicator(), y.communicator(), &compare);
    assert( compare == MPI_CONGRUENT || compare == MPI_IDENT); 
#endif //DG_DEBUG
    typedef typename Vector::container_type container;
    //local compuation
    exblas::Superaccumulator acc_fine = doDot_dispatch( x.data(), y.data(),typename VectorTraits<container>::vector_category()), acc_reduce;  
    acc_fine.Normalize();
    //communication (we cannot sum more than 128 accumulators at once, so we need to split)
    std::vector<int64_t> receive(39,0);
    MPI_Reduce(&(acc_fine.get_accumulator()[0]), &(receive[0]), acc_fine.get_f_words() + acc_fine.get_e_words(), MPI_LONG, MPI_SUM, 0, x.communicator_mod()); 
    int rank;
    MPI_Comm_rank( x.communicator_mod(), &rank);
    if(x.communicator_mod_reduce() != MPI_COMM_NULL)
    {
        exblas::Superaccumulator acc_reduce( receive);
        acc_reduce.Normalize();
        receive.assign(39,0);
        MPI_Reduce(&(acc_reduce.get_accumulator()[0]), &(receive[0]), acc_fine.get_f_words() + acc_fine.get_e_words(), MPI_LONG, MPI_SUM, 0, x.communicator_mod_reduce()); 
    }
    MPI_Bcast( &(receive[0]), 39, MPI_LONG, 0, x.communicator());

    exblas::Superaccumulator result(receive);
    return result.Round();
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
