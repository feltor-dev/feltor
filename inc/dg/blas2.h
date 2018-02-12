#pragma once

#include "backend/vector_traits.h"
#include "backend/matrix_traits.h"
#include "backend/cusp_precon_blas.h"
#include "backend/matrix_traits_thrust.h"
#include "backend/thrust_matrix_blas.cuh"
#include "backend/cusp_matrix_blas.cuh"
#include "backend/sparseblockmat.cuh"
#include "backend/selfmade_blas.cuh"
#ifdef MPI_VERSION
#include "backend/mpi_matrix_blas.h"
#include "backend/mpi_precon_blas.h"
#endif //MPI_VERSION
#include "backend/std_matrix_blas.cuh"


/*!@file
 *
 * Basic linear algebra level 2 functions (functions that involve vectors and matrices)
 */
namespace dg{
/*! @brief BLAS Level 2 routines

 @ingroup blas2
 @note Only those routines that are actually called need to be implemented for a given type.
*/
namespace blas2{
///@addtogroup blas2
///@{

/**
 * @brief \f$ y = x\f$; Generic way to copy and/or convert a Matrix type to a different Matrix type (e.g. from CPU to GPU, or double to float, etc.)
 *
 * @copydoc hide_matrix
 * @tparam AnotherMatrix Another Matrix type
 * @param x source
 * @param y sink
 * @note y gets resized properly
 * @copydoc hide_code_blas2_symv
 */
template<class Matrix, class AnotherMatrix>
inline void transfer( const Matrix& x, AnotherMatrix& y)
{
    dg::blas2::detail::doTransfer( x,y, get_matrix_category<Matrix>(), get_matrix_category<AnotherMatrix>());
}

/*! @brief \f$ x^T M y\f$; Binary reproducible general dot product
 *
 * This routine computes the scalar product defined by the symmetric positive definite
 * matrix M \f[ x^T M y = \sum_{i,j=0}^{N-1} x_i M_{ij} y_j \f]
 * ( Note that if M is not diagonal it is generally more efficient to
 * precalculate \f$ My\f$ and then call the \c dg::blas1::dot() routine!
 * Our implementation guarantees binary reproducible results up to and excluding the last mantissa bit of the result.
 * Furthermore, the sum is computed with infinite precision and the result is then rounded
 * to the nearest double precision number. Although the products are not computed with
 * infinite precision, the order of multiplication is guaranteed.
 * This is possible with the help of an adapted version of the \c ::exblas library.
 * @tparam DiagonalMatrix Right now \c DiagonalMatrix has to be the same as \c container, except if \c container is a \p std::vector<container_type>, then the \c DiagonalMatrix has to be the \c container_type.
 * In the latter case the Matrix is applied to all containers in the std::vector and the sum is returned.
 * @copydoc hide_container
 * @param x Left container
 * @param m The diagonal Matrix
 * @param y Right container (may alias \p x)
 * @return Generalized scalar product
 * @note This routine is always executed synchronously due to the
    implicit memcpy of the result.
 * @copydoc hide_code_evaluate2d
 */
template< class DiagonalMatrix, class container>
inline typename MatrixTraits<DiagonalMatrix>::value_type dot( const container& x, const DiagonalMatrix& m, const container& y)
{
    return dg::blas2::detail::doDot( x, m, y,
                       get_matrix_category<DiagonalMatrix>(),
                       get_vector_category<container>() );
}

/*! @brief \f$ x^T M x\f$; Binary reproducible general dot product
 *
 * \f[ x^T M x = \sum_{i,j=0}^{N-1} x_i M_{ij} x_j \f]
 * @tparam DiagonalMatrix Right now \c DiagonalMatrix has to be the same as \c container, except if \c container is a \c std::vector<container_type>, then the \c DiagonalMatrix has to be the \c container_type.
 * In the latter case the Matrix is applied to all containers in the std::vector and the sum is returned.
 * @copydoc hide_container
 * @param m The diagonal Matrix
 * @param x Right container
 * @return Generalized scalar product
 * @note This routine is always executed synchronously due to the
    implicit memcpy of the result.
 * @note This routine is equivalent to the call \c dg::blas2::dot( x, m, x);
     which should be prefered because it looks more explicit
 */
template< class DiagonalMatrix, class container>
inline typename MatrixTraits<DiagonalMatrix>::value_type dot( const DiagonalMatrix& m, const container& x)
{
    return dg::blas2::detail::doDot( m, x,
                       get_matrix_category<DiagonalMatrix>(),
                       get_vector_category<container>() );
}

/*! @brief \f$ y = \alpha M x + \beta y\f$
 *
 * This routine computes \f[ y = \alpha M x + \beta y \f]
 * where \f$ M\f$ is a matrix.
 * @copydoc hide_matrix
 * @copydoc hide_container
 * @param alpha A Scalar
 * @param M The Matrix
 * @param x A container different from \p y
 * @param beta A Scalar
 * @param y contains the solution on output (may not alias \p x)
 * @attention \p y may never alias \p x
 * @copydoc hide_code_blas2_symv
 */
template< class Matrix, class container>
inline void symv( get_value_type<container> alpha,
                  const Matrix& M,
                  const container& x,
                  get_value_type<container> beta,
                  container& y)
{
    if(alpha == (get_value_type<container>)0) {
        dg::blas1::scal( y, alpha);
        return;
    }
    dg::blas2::detail::doSymv( alpha, M, x, beta, y,
                       get_matrix_category<Matrix>(),
                       get_vector_category<container>() );
    return;
}

/*! @brief \f$ y = M x\f$
 *
 * This routine computes \f[ y = M x \f]
 * where \f$ M\f$ is a matrix.
 * @copydoc hide_matrix
 * @copydoc hide_container
 * @tparam same_or_another_container Currently needs to be the same as \c container.
 * @param M The Matrix
 * @param x A container different from \p y
 * @param y contains the solution on output (may not alias \p x)
 * @attention y may never alias x
 * @note Due to the \c SelfMadeMatrixTag, M cannot be declared const
 * @copydoc hide_code_blas2_symv
 */
template< class Matrix, class container, class same_or_another_container>
inline void symv( Matrix& M,
                  const container& x,
                  same_or_another_container& y)
{
    dg::blas2::detail::doSymv( M, x, y,
                       get_matrix_category<Matrix>(),
                       get_vector_category<container>(),
                       typename dg::VectorTraits<same_or_another_container>::vector_category() );
    return;
}

/*! @brief \f$ y = M x\f$;
 * General Matrix-Vector product
 *
 * Does exactly the same as symv.
 * @copydoc hide_matrix
 * @copydoc hide_container
 * @tparam same_or_another_container Currently needs to be the same as \c container.
 * @param M The Matrix
 * @param x A container different from \p y
 * @param y contains the solution on output (may not alias \p x)
 * @attention y may never alias \p x
 */
template< class Matrix, class container, class same_or_another_container>
inline void gemv( Matrix& M,
                  const container& x,
                  same_or_another_container& y)
{
    dg::blas2::detail::doGemv( M, x, y,
                       get_matrix_category<Matrix>(),
                       get_vector_category<container>(),
                       typename dg::VectorTraits<same_or_another_container>::vector_category() );
    return;
}
/*! @brief \f$ y = \alpha M x + \beta y \f$;
 * General Matrix-Vector product
 *
 * Does exactly the same as symv.
 * @copydoc hide_matrix
 * @copydoc hide_container
 * @param alpha A Scalar
 * @param M The Matrix
 * @param x A container different from \p y
 * @param beta A Scalar
 * @param y contains the solution on output (may not alias \p x)
 * @attention y may never alias \p x
 */
template< class Matrix, class container>
inline void gemv( get_value_type<container> alpha,
                  const Matrix& M,
                  const container& x,
                  get_value_type<container> beta,
                  container& y)
{
    if(alpha == (get_value_type<container>)0) {
        dg::blas1::scal( y, alpha);
        return;
    }
    dg::blas2::detail::doGemv( alpha, M, x, beta, y,
                       get_matrix_category<Matrix>(),
                       get_vector_category<container>() );
    return;
}
///@}

} //namespace blas2
} //namespace dg
