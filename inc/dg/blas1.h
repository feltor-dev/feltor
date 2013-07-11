#ifndef _DG_BLAS1_
#define _DG_BLAS1_

#include "vector_traits.h"
#include "blas/thrust_vector.cuh"
#include "blas/std_vector.cuh"

namespace dg{

//TODO Throw messages instead of assertions because origin of message can 
// be better followed
//
// eigentlich dürften die blas routinen nicht im dg namensraum liegen, denn damit 
// haben sie erst mal nichts zu tun. Wenn man nun als Außenstehender seine eigene 
// Vektorklasse hat und die blas spezialisieren will? (Bis auf cudaThreadSynchronize natürlich
// aber das kann man ja auch templaten, schließlich gibts auch noch openMP, MPI etc.)
// IDEE: synchronize<Vector>();
/*! @brief BLAS Level 1 routines 
 *
 * @ingroup blas1
 * Only those routines that are actually called need to be implemented.
 * Don't forget to specialize in the dg namespace.
 * @note successive calls to blas routines are executed sequentially 
 */
namespace blas1
{

///@addtogroup blas1
///@{

/*! @brief Euclidean dot product between two Vectors
 *
 * This routine computes \f[ x^T y = \sum_{i=0}^{N-1} x_i y_i \f]
 * @param x Left Vector
 * @param y Right Vector might equal y
 * @return Scalar product
 * @note This routine is always executed synchronously due to the 
        implicit memcpy of the result.
 * @note If DG_DEBUG is defined a range check shall be performed
 */
template< class Vector>
inline typename Vector::value_type dot( const Vector& x, const Vector& y)
{
    return dg::blas1::detail::doDot( x, y, typename dg::VectorTraits<Vector>::vector_category() );
}

/*! @brief Modified BLAS 1 routine axpy
 *
 * This routine computes \f[ y =  \alpha x + \beta y \f] 
 * Q: Isn't it better to implement daxpy and daypx? \n
 * A: unlikely, because in all three cases all elements of x and y have to be loaded
 * and daxpy is memory bound. (Is there no original daxpby routine because 
 * the name is too long??)
 * @param alpha Scalar  
 * @param x Vector x might equal y 
 * @param beta Scalar
 * @param y Vector y contains solution on output
 * @note In an implementation you may want to check for alpha == 0 and beta == 1
 * @note If DG_DEBUG is defined a range check shall be performed
 * @attention If a thrust::device_vector ist used then this routine is NON-BLOCKING!
 */
template< class Vector>
inline void axpby( typename VectorTraits<Vector>::value_type alpha, const Vector& x, typename VectorTraits<Vector>::value_type beta, Vector& y)
{
    return dg::blas1::detail::doAxpby( alpha, x, beta, y, typename dg::VectorTraits<Vector>::vector_category() );
}

/*! @brief Modified BLAS 1 routine axpy
 *
 * This routine computes \f[ z =  \alpha x + \beta y \f] 
 * @param alpha Scalar  
 * @param x Vector x might equal result
 * @param beta Scalar
 * @param y Vector y might equal result
 * @param result Vector contains solution on output
 * @note In an implementation you may want to check for alpha == 0 and beta == 1
 * @note If DG_DEBUG is defined a range check shall be performed
 * @attention If a thrust::device_vector is used then this routine is NON-BLOCKING!
 */
template< class Vector>
inline void axpby( typename VectorTraits<Vector>::value_type alpha, const Vector& x, typename VectorTraits<Vector>::value_type beta, const Vector& y, Vector& result)
{
    return dg::blas1::detail::doAxpby( alpha, x, beta, y, result, typename dg::VectorTraits<Vector>::vector_category() );
}

/**
* @brief A 'new' BLAS 1 routine. 
*
* Multiplies two vectors element by element: \f[ y_i = x_{1i}x_{2i}\f]
* @param x1 Vector x1  
* @param x2 Vector x2 might equal x1
* @param y  Vector y contains result on output ( might equal x1 or x2)
* @note If DG_DEBUG is defined a range check shall be performed 
*/
template< class Vector>
inline void pointwiseDot( const Vector& x1, const Vector& x2, Vector& y)
{
    return dg::blas1::detail::doPointwiseDot( x1, x2, y, typename dg::VectorTraits<Vector>::vector_category() );
}
///@}
}//namespace blas1
} //namespace dg


#endif //_DG_BLAS1_
