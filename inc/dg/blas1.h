#pragma once

#include "backend/vector_traits.h"
#ifdef MPI_VERSION
#include "backend/mpi_vector.h"
#include "backend/mpi_vector_blas.h"
#endif
#include "backend/thrust_vector.cuh"
#include "backend/std_vector.cuh"

/*!@file 
 *
 * blas level 1 functions
 */
namespace dg{

//TODO Throw messages instead of assertions because origin of message can 
// be better followed ?
//
// eigentlich dürften die blas routinen nicht im dg namensraum liegen, denn damit 
// haben sie erst mal nichts zu tun. Wenn man nun als Außenstehender seine eigene 
// Vektorklasse hat und die blas spezialisieren will?
// Synchronize ist niemals nötig mit thrust!!
// Vielleicht wäre function overloading die bessere design - Wahl für blas
/*! @brief BLAS Level 1 routines 
 *
 * @ingroup blas1
 * Only those routines that are actually called need to be implemented.
 * Don't forget to specialize in the dg namespace.
 * @note successive calls to blas routines are executed sequentially 
 * @note A manual synchronization of threads or devices is never needed in an application 
 * using these functions. All functions returning a value block until the value is ready.
 */
namespace blas1
{

///@addtogroup blas1
///@{

/*! @brief Euclidean dot product between two Vectors
 *
 * This routine computes \f[ x^T y = \sum_{i=0}^{N-1} x_i y_i \f]
 * @param x Left Vector
 * @param y Right Vector may equal y
 * @return Scalar product
 * @note This routine is always executed synchronously due to the 
        implicit memcpy of the result. With mpi the result is broadcasted to all
        processes
 * @note If DG_DEBUG is defined a range check shall be performed
 */
template< class Vector>
inline typename VectorTraits<Vector>::value_type dot( const Vector& x, const Vector& y)
{
    return dg::blas1::detail::doDot( x, y, typename dg::VectorTraits<Vector>::vector_category() );
}

/*! @brief Modified BLAS 1 routine axpy
 *
 * This routine computes \f[ y_i =  \alpha x_i + \beta y_i \f] 
 * @param alpha Scalar  
 * @param x Vector x may equal y 
 * @param beta Scalar
 * @param y Vector y contains solution on output
 * @note Checks for alpha == 0 and beta == 1
 * @note If DG_DEBUG is defined a range check shall be performed
 */
template< class Vector>
inline void axpby( typename VectorTraits<Vector>::value_type alpha, const Vector& x, typename VectorTraits<Vector>::value_type beta, Vector& y)
{
    dg::blas1::detail::doAxpby( alpha, x, beta, y, typename dg::VectorTraits<Vector>::vector_category() );
    return;
}

/*! @brief Modified BLAS 1 routine axpy
 *
 * This routine computes \f[ z_i =  \alpha x_i + \beta y_i \f] 
 * @param alpha Scalar  
 * @param x Vector x may equal result
 * @param beta Scalar
 * @param y Vector y may equal result
 * @param result Vector contains solution on output
 * @note Checks for alpha == 0 and beta == 1
 * @note If DG_DEBUG is defined a range check shall be performed
 * @attention If a thrust::device_vector is used then this routine is NON-BLOCKING!
 */
template< class Vector>
inline void axpby( typename VectorTraits<Vector>::value_type alpha, const Vector& x, typename VectorTraits<Vector>::value_type beta, const Vector& y, Vector& result)
{
    dg::blas1::detail::doAxpby( alpha, x, beta, y, result, typename dg::VectorTraits<Vector>::vector_category() );
    return;
}
/*! @brief "new" BLAS 1 routine transform
 *
 * This routine computes \f[ y_i = f(x_i) \f] 
 * This is actually not a BLAS routine since f can be a nonlinear function.
 * It is rather the first step towards a more general library conception.
 * @param x Vector x may equal y
 * @param y Vector y contains result, may equal x
 * @param op unary Operator to use on every element
 * @note In an implementation you may want to check for alpha == 0
 */
template< class Vector, class UnaryOp>
inline void transform( const Vector& x, Vector& y, UnaryOp op)
{
    dg::blas1::detail::doTransform( x, y, op, typename dg::VectorTraits<Vector>::vector_category() );
    return;
}

/*! @brief BLAS 1 routine scal
 *
 * This routine computes \f[ \alpha x_i \f] 
 * @param alpha Scalar  
 * @param x Vector x 
 */
template< class Vector>
inline void scal( Vector& x, typename VectorTraits<Vector>::value_type alpha)
{
    dg::blas1::detail::doScal( x, alpha, typename dg::VectorTraits<Vector>::vector_category() );
    return;
}

/**
* @brief A 'new' BLAS 1 routine. 
*
* Multiplies two vectors element by element: \f[ y_i = x_{1i}x_{2i}\f]
* @param x1 Vector x1  
* @param x2 Vector x2 may equal x1
* @param y  Vector y contains result on output ( may equal x1 or x2)
* @note If DG_DEBUG is defined a range check shall be performed 
*/
template< class Vector>
inline void pointwiseDot( const Vector& x1, const Vector& x2, Vector& y)
{
    dg::blas1::detail::doPointwiseDot( x1, x2, y, typename dg::VectorTraits<Vector>::vector_category() );
    return;
}
/**
* @brief A 'new' BLAS 1 routine. 
*
* Divides two vectors element by element: \f[ y_i = x_{1i}/x_{2i}\f]
* @param x1 Vector x1  
* @param x2 Vector x2 may equal x1
* @param y  Vector y contains result on output ( ma equal x1 and/or x2)
* @note If DG_DEBUG is defined a range check shall be performed 
*/
template< class Vector>
inline void pointwiseDivide( const Vector& x1, const Vector& x2, Vector& y)
{
    dg::blas1::detail::doPointwiseDivide( x1, x2, y, typename dg::VectorTraits<Vector>::vector_category() );
    return;
}
///@}
}//namespace blas1
} //namespace dg

