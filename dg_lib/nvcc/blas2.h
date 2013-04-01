#ifndef _DG_BLAS2_
#define _DG_BLAS2_

#include "blas/cusp_matrix.cuh"
#include "blas/preconditioner.cuh"
#include "blas/operator.cuh"
#include "vector_traits.h"
#include "matrix_traits.h"

namespace dg{
/*! @brief BLAS Level 2 routines 

 @ingroup blas
 In an implementation Vector and Matrix should be typedefed.
 Only those routines that are actually called need to be implemented.
*/
namespace blas2{

    /*! @brief General dot produt
     *
     * This routine computes the scalar product defined by the symmetric positive definit 
     * matrix P \f[ x^T P y = \sum_{i=0}^{N-1} x_i P_{ij} y_j \f]
     * where P is a diagonal matrix. (Otherwise it would be more efficient to 
     * precalculate \f[ Py\f] and then call the BLAS1::dot routine!
     * @param x Left Vector
     * @param P The diagonal Matrix
     * @param y Right Vector might equal Left Vector
     * @return Generalized scalar product
     * @note This routine is always executed synchronously due to the 
        implicit memcpy of the result.
     */
template< class Matrix, class Vector>
inline typename Matrix::value_type dot( const Vector& x, const Matrix& m, const Vector& y)
{
    return dg::blas2::detail::doDot( x, m, y, 
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<Vector>::vector_category() );
}

/*! @brief General dot produt
 *
 * This routine is equivalent to the call dot( x, P, x)
 * @param P The diagonal Matrix
 * @param x Right Vector
 * @return Generalized scalar product
 * @note This routine is always executed synchronously due to the 
    implicit memcpy of the result.
 */
template< class Matrix, class Vector>
inline typename Matrix::value_type dot( const Matrix& m, const Vector& x)
{
    return dg::blas2::detail::doDot( m, x, 
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<Vector>::vector_category() );
}

/*! @brief Symmetric Matrix Vector product
 *
 * This routine computes \f[ y = \alpha M x + \beta y \f]
 * where \f[ M\f] is a symmetric matrix. 
 * @param alpha A Scalar
 * @param m The Matrix
 * @param x A Vector different from y (except in the case where m is diagonal)
 * @param beta A Scalar
 * @param y contains solution on output
 * @note In an implementation you may want to check for alpha == 0 and beta == 1
 * @attention If a thrust::device_vector ist used then this routine is NON-BLOCKING!
 */
template< class Matrix, class Vector>
inline void symv( typename MatrixTraits<Matrix>::value_type alpha, 
                  const Matrix& m, 
                  const Vector& x, 
                  typename MatrixTraits<Matrix>::value_type beta, 
                  Vector& y)
{
    return dg::blas2::detail::doSymv( alpha, m, x, beta, y, 
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<Vector>::vector_category() );
}

/*! @brief Symmetric Matrix Vector product
 *
 * This routine is equivalent to dsymv( 1., m, x, 0., y);
 * @param m The Matrix
 * @param x A Vector different from y (except in the case where m is diagonal)
 * @param y contains solution on output
 * @attention If a thrust::device_vector ist used then this routine is NON-BLOCKING!
 */
template< class Matrix, class Vector>
inline void symv( const Matrix& m, 
                  const Vector& x, 
                  Vector& y)
{
    return dg::blas2::detail::doSymv( m, x, y, 
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<Vector>::vector_category() );
}

} //namespace blas2
} //namespace dg

#endif //_DG_BLAS2_
