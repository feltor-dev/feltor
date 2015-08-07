#pragma once 

#ifdef MPI_VERSION
//#include "backend/mpi_matrix_blas.h"
#include "backend/mpi_precon_blas.h"
#endif //MPI_VERSION
#include "backend/vector_traits.h"
#include "backend/matrix_traits.h"
#include "backend/matrix_traits_thrust.h"
#include "backend/thrust_matrix.cuh"
#include "backend/cusp_matrix.cuh"
#include "backend/sparseblockmat.cuh"
#include "backend/selfmade.cuh"
#include "backend/std_matrix.cuh"


/*!@file 
 *
 * blas level 2 functions
 */
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
 * matrix M \f[ x^T M y = \sum_{i,j=0}^{N-1} x_i M_{ij} y_j \f]
 * ( Note that if M is not diagonal it is generally more efficient to 
 * precalculate \f$ My\f$ and then call the blas1::dot() routine!
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
 * This routine is equivalent to the call blas2::dot( x, m, x):
 * \f[ x^T M x = \sum_{i,j=0}^{N-1} x_i M_{ij} x_j \f]
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
 * This routine computes \f[ y = \alpha P x + \beta y \f]
 * where \f$ P\f$ is a symmetric Preconditioner. 
 * P should be diagonal since
 * otherwise a call to symv() followed by axpby() is faster.
 * @param alpha A Scalar
 * @param P The Preconditioner
 * @param x A Vector different from y (except in the case where m is diagonal)
 * @param beta A Scalar
 * @param y contains solution on output
 */
template< class Precon, class Vector>
inline void symv( typename MatrixTraits<Precon>::value_type alpha, 
                  const Precon& P, 
                  const Vector& x, 
                  typename MatrixTraits<Precon>::value_type beta, 
                  Vector& y)
{
    dg::blas2::detail::doSymv( alpha, P, x, beta, y, 
                       typename dg::MatrixTraits<Precon>::matrix_category(), 
                       typename dg::VectorTraits<Vector>::vector_category() );
    return;
}

/*! @brief Symmetric Matrix Vector product
 *
 * This routine computes \f[ y = M x \f]
 * where \f$ M\f$ is a symmetric matrix. 
 * @param m The Matrix
 * @param x A Vector different from y (except in the case where m is diagonal)
 *      In most applications x is assumed to remain constant. 
 * @param y contains solution on output
 * @attention Due to the SelfMadeMatrixTag and MPI_Vectors, m and x cannot be declared const
 */
template< class Matrix, class Vector1, class Vector2>
inline void symv( Matrix& m, 
                  Vector1& x, 
                  Vector2& y)
{
    dg::blas2::detail::doSymv( m, x, y, 
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<Vector1>::vector_category(),
                       typename dg::VectorTraits<Vector2>::vector_category() );
    return;
}
///@cond
template< class Matrix, class Vector>
inline void mv(   Matrix& m, 
                  const Vector& x, 
                  Vector& y)
{
    dg::blas2::detail::doSymv( m, x, y, 
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<const Vector>::vector_category(),
                       typename dg::VectorTraits<Vector>::vector_category() );
    return;
}
///@endcond

/**
 * @brief General Matrix-Vector product
 *
 * @param m The Matrix
 * @param x A Vector different from y 
 * @param y contains the solution on output
 */
template< class Matrix, class Vector1, class Vector2>
inline void gemv( Matrix& m, 
                  Vector1& x, 
                  Vector2& y)
{
    dg::blas2::detail::doGemv( m, x, y, 
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<Vector1>::vector_category(),
                       typename dg::VectorTraits<Vector2>::vector_category() );
    return;
}
///@}

} //namespace blas2
} //namespace dg

