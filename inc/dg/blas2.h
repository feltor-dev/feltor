#pragma once 

#include "vector_traits.h"
#include "matrix_traits.h"
#include "matrix_traits_thrust.h"
#include "blas/thrust_matrix.cuh"
#include "blas/std_matrix.cuh"
#include "blas/cusp_matrix.cuh"
#include "blas/selfmade.cuh"
#ifdef MPI_BACKEND
#include "blas/mpi_matrix.h"
#endif //MPI_BACKEND

namespace dg{
/*! @brief BLAS Level 2 routines 

 @ingroup blas2
 In an implementation Vector and Matrix should be typedefed.
 Only those routines that are actually called need to be implemented.
*/
namespace blas2{
///@addtogroup blas2
///@{

/*! @brief General dot produt
 *
 * This routine computes the scalar product defined by the symmetric positive definite 
 * matrix M \f[ x^T M y = \sum_{i=0}^{N-1} x_i M_{ij} y_j \f]
 * ( Note that if M is not diagonal it is generally more efficient to 
 * precalculate \f[ My\f] and then call the BLAS1::dot routine!
 * @param x Left Vector
 * @param m The diagonal Matrix
 * @param y Right Vector might equal Left Vector
 * @return Generalized scalar product
 * @note This routine is always executed synchronously due to the 
    implicit memcpy of the result.
 */
template< class Matrix, class Vector>
inline typename MatrixTraits<Matrix>::value_type dot( const Vector& x, const Matrix& m, const Vector& y)
{
    return dg::blas2::detail::doDot( x, m, y, 
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<Vector>::vector_category() );
}

/*! @brief General dot produt
 *
 * This routine is equivalent to the call dot( x, m, x)
 * @param m The diagonal Matrix
 * @param x Right Vector
 * @return Generalized scalar product
 * @note This routine is always executed synchronously due to the 
    implicit memcpy of the result.
 */
template< class Matrix, class Vector>
inline typename MatrixTraits<Matrix>::value_type dot( const Matrix& m, const Vector& x)
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
 * This routine computes \f[ y = M x \f]
 * where \f[ M\f] is a symmetric matrix. 
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
///@cond
template< class Matrix, class Vector>
inline void mv( const Matrix& m, 
                  const Vector& x, 
                  Vector& y)
{
    return dg::blas2::detail::doSymv( m, x, y, 
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<Vector>::vector_category() );
}
///@endcond

/**
 * @brief General Matrix-Vector product
 *
 * @param m The Matrix
 * @param x A Vector different from y 
 * @param y contains the solution on output
 * @attention If a thrust::device_vector ist used then this routine is NON-BLOCKING!
 */
template< class Matrix, class Vector>
inline void gemv( const Matrix& m, 
                  const Vector& x, 
                  Vector& y)
{
    return dg::blas2::detail::doGemv( m, x, y, 
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<Vector>::vector_category() );
}
///@}

} //namespace blas2
} //namespace dg

