#ifndef _DG_BLAS_
#define _DG_BLAS_

#include "vector_traits.h"
#include "blas/thrust_vector.cuh"


namespace dg{

//CUDA relevant: BLAS routines must block until result is ready 
/*! @brief BLAS Level 1 routines 
 *
 * @ingroup blas
 * Only those routines that are actually called need to be implemented.
 * Don't forget to specialize in the dg namespace
 */
namespace blas1
{

/*! @brief Euclidean dot product between two Vectors
 *
 * This routine computes \f[ x^T y = \sum_{i=0}^{N-1} x_i y_i \f]
 * @param x Left Vector
 * @param y Right Vector might equal Left Vector
 * @return Scalar product
 * @note This routine is always executed synchronously due to the 
        implicit memcpy of the result.
 */
template< class Vector>
inline typename Vector::value_type dot( const Vector& x, const Vector& y)
{
    return dg::blas1::detail::doDot( x, y, typename dg::VectorTraits<Vector>::vector_category() );
}

/*! @brief Modified BLAS 1 routine daxpy
 *
 * This routine computes \f[ y =  \alpha x + \beta y \f] 
 * Q: Isn't it better to implement daxpy and daypx? \n
 * A: unlikely, because in all three cases all elements of x and y have to be loaded
 * and daxpy is memory bound. (Is there no original daxpby routine because 
 * the name is too long??)
 * @param alpha Scalar  
 * @param x Vector x migtht equal y 
 * @param beta Scalar
 * @param y Vector y contains solution on output
 * @note In an implementation you may want to check for alpha == 0 and beta == 1
 * @attention If a thrust::device_vector ist used then this routine is NON-BLOCKING!
 */
template< class Vector>
inline void axpby( typename Vector::value_type alpha, const Vector& x, typename Vector::value_type beta, Vector& y)
{
    return dg::blas1::detail::doAxpby( alpha, x, beta, y, typename dg::VectorTraits<Vector>::vector_category() );
}

}//namespace blas1
} //namespace dg

#include "blas2.h"

#endif //_DG_BLAS_
