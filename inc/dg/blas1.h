#pragma once

#include "backend/vector_traits.h"
#include "backend/thrust_vector_blas.cuh"
#include "backend/cusp_vector_blas.h"
#ifdef MPI_VERSION
#include "backend/mpi_vector.h"
#include "backend/mpi_vector_blas.h"
#endif
#include "backend/std_vector_blas.cuh"

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

/**
 * @brief Generic way to copy vectors of different types (e.g. from CPU to GPU, or double to float, etc.)
 *
 * @tparam Vector1 First vector type
 * @tparam Vector2 Second vector type
 * @param x source
 * @param y sink
 * @note y gets resized properly
 */
template<class Vector1, class Vector2>
inline void transfer( const Vector1& x, Vector2& y)
{
    dg::blas1::detail::doTransfer( x,y, typename dg::VectorTraits<Vector1>::vector_category(), typename dg::VectorTraits<Vector2>::vector_category());
}


/**
 * @brief Invoke assignment operator
 *
 * @tparam Vector Vector class
 * @param x in
 * @param y out
 */
template<class Vector>
inline void copy( const Vector& x, Vector& y){y=x;}

/*! @brief Euclidean dot product between two Vectors
 *
 * This routine computes \f[ x^T y = \sum_{i=0}^{N-1} x_i y_i \f]
 * @param x Left Vector
 * @param y Right Vector may equal x
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

/*! @brief Modified BLAS 1 routine axpy
 *
 * This routine computes \f[ z_i =  \alpha x_i + \beta y_i + \gamma z_i \f] 
 * @param alpha Scalar  
 * @param x Vector x may equal result
 * @param beta Scalar
 * @param y Vector y may equal result
 * @param gamma Scalar
 * @param z Vector contains solution on output
 * @note If DG_DEBUG is defined a range check shall be performed
 * @attention If a thrust::device_vector is used then this routine is NON-BLOCKING!
 */
template< class Vector>
inline void axpbygz( typename VectorTraits<Vector>::value_type alpha, const Vector& x, typename VectorTraits<Vector>::value_type beta, const Vector& y, typename VectorTraits<Vector>::value_type gamma, Vector& z)
{
    dg::blas1::detail::doAxpby( alpha, x, beta, y, gamma, z, typename dg::VectorTraits<Vector>::vector_category() );
    return;
}

/*! @brief "new" BLAS 1 routine transform
 *
 * This routine computes \f[ y_i = op(x_i) \f] 
 * This is strictly speaking not a BLAS routine since f can be a nonlinear function.
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

/*! @brief pointwise add a scalar
 *
 * This routine computes \f[ x_i + \alpha \f] 
 * @param alpha Scalar  
 * @param x Vector x 
 */
template< class Vector>
inline void plus( Vector& x, typename VectorTraits<Vector>::value_type alpha)
{
    dg::blas1::detail::doPlus( x, alpha, typename dg::VectorTraits<Vector>::vector_category() );
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
* Multiplies two vectors element by element: \f[ y_i = \alpha x_{1i}x_{2i} + \beta y_i\f]
* @param alpha scalar
* @param x1 Vector x1  
* @param x2 Vector x2 may equal x1
* @param beta scalar
* @param y  Vector y contains result on output ( may equal x1 or x2)
* @note If DG_DEBUG is defined a range check shall be performed 
*/
template< class Vector>
inline void pointwiseDot( typename VectorTraits<Vector>::value_type alpha, const Vector& x1, const Vector& x2, typename VectorTraits<Vector>::value_type beta, Vector& y)
{
    dg::blas1::detail::doPointwiseDot( alpha, x1, x2, beta, y, typename dg::VectorTraits<Vector>::vector_category() );
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

/**
* @brief A 'new' fused multiply-add BLAS 1 routine. 
*
* Multiplies and adds vectors element by element: \f[ z_i = \alpha x_{1i}y_{1i} + \beta x_{2i]y_{2i} + \gamma z_i\f]
* @param alpha scalar
* @param x1 Vector x1  
* @param y1 Vector y1 
* @param beta scalar
* @param x2 Vector x1  
* @param y2 Vector y1 
* @param z  Vector z contains result on output 
* @note aliases are allowed: we perform an alias analysis on the given references to detect possible performance optimizations
*/
template<class Vector>
void pointwiseDot(  typename VectorTraits<Vector>::value_type alpha, const Vector& x1, const Vector& y1, 
                    typename VectorTraits<Vector>::value_type beta,  const Vector& x2, const Vector& y2, 
                    typename VectorTraits<Vector>::value_type gamma, Vector & z)
{
    dg::blas1::detail::doPointwiseDot( alpha, x1, y1, beta, x2, y2, gamma, z, typename dg::VectorTraits<Vector>::vector_category() );
    return;
}
///@}
}//namespace blas1
} //namespace dg

