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
 * @brief Generic way to copy and/or convert a Matrix type to a different Matrix type (e.g. from CPU to GPU, or double to float, etc.)
 *
 * @tparam Matrix First Matrix type
 * @tparam AnotherMatrix Second Matrix type
 * @param x source
 * @param y sink 
 * @note y gets resized properly
 */
template<class Matrix, class AnotherMatrix>
inline void transfer( const Matrix& x, AnotherMatrix& y)
{
    dg::blas2::detail::doTransfer( x,y, typename dg::MatrixTraits<Matrix>::matrix_category(), typename dg::MatrixTraits<AnotherMatrix>::matrix_category());
}

/*! @brief General dot produt
 *
 * This routine computes the scalar product defined by the symmetric positive definite 
 * matrix M \f[ x^T M y = \sum_{i,j=0}^{N-1} x_i M_{ij} y_j \f]
 * ( Note that if M is not diagonal it is generally more efficient to 
 * precalculate \f$ My\f$ and then call the blas1::dot() routine!
 * @tparam DiagonalMatrix Right now DiagonalMatrix has to be the same as container, except if the container is a std::vector<container_type>, then the DiagonalMatrix has to be the container_type
 * @copydoc hide_container_lvl1
 * @param x Left container
 * @param m The diagonal Matrix
 * @param y Right container might equal Left container
 * @return Generalized scalar product
 * @note This routine is always executed synchronously due to the 
    implicit memcpy of the result.
 */
template< class DiagonalMatrix, class container>
inline typename MatrixTraits<DiagonalMatrix>::value_type dot( const container& x, const DiagonalMatrix& m, const container& y)
{
    return dg::blas2::detail::doDot( x, m, y, 
                       typename dg::MatrixTraits<DiagonalMatrix>::matrix_category(), 
                       typename dg::VectorTraits<container>::vector_category() );
}

/*! @brief General dot produt
 *
 * This routine is equivalent to the call blas2::dot( x, m, x):
 * \f[ x^T M x = \sum_{i,j=0}^{N-1} x_i M_{ij} x_j \f]
 * @param m The diagonal Matrix
 * @param x Right container
 * @return Generalized scalar product
 * @note This routine is always executed synchronously due to the 
    implicit memcpy of the result.
 */
template< class Matrix, class container>
inline typename MatrixTraits<Matrix>::value_type dot( const Matrix& m, const container& x)
{
    return dg::blas2::detail::doDot( m, x, 
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<container>::vector_category() );
}

/*! @brief Symmetric Matrix container product
 *
 * This routine computes \f[ y = \alpha P x + \beta y \f]
 * where \f$ P\f$ is a symmetric Preconditioner. 
 * @param alpha A Scalar
 * @param P The Preconditioner
 * @param x A container different from y (except in the case where m is diagonal)
 * @param beta A Scalar
 * @param y contains solution on output
 */
template< class Precon, class container>
inline void symv( typename MatrixTraits<Precon>::value_type alpha, 
                  const Precon& P, 
                  const container& x, 
                  typename MatrixTraits<Precon>::value_type beta, 
                  container& y)
{
    if(alpha == (typename MatrixTraits<Precon>::value_type)0) {
        dg::blas1::scal( y, alpha);
        return;
    }
    dg::blas2::detail::doSymv( alpha, P, x, beta, y, 
                       typename dg::MatrixTraits<Precon>::matrix_category(), 
                       typename dg::VectorTraits<container>::vector_category() );
    return;
}

/*! @brief Symmetric Matrix container product
 *
 * This routine computes \f[ y = M x \f]
 * where \f$ M\f$ is a symmetric matrix. 
 * @param m The Matrix
 * @param x A container different from y (except in the case where m is diagonal)
 *      In most applications x is assumed to remain constant. 
 * @param y contains solution on output
 * @attention Due to the SelfMadeMatrixTag and MPI_containers, m and x cannot be declared const
 */
template< class Matrix, class container1, class container2>
inline void symv( Matrix& m, 
                  container1& x, 
                  container2& y)
{
    dg::blas2::detail::doSymv( m, x, y, 
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<container1>::vector_category(),
                       typename dg::VectorTraits<container2>::vector_category() );
    return;
}

/**
 * @brief General Matrix-container product
 *
 * @param m The Matrix
 * @param x A container different from y 
 * @param y contains the solution on output
 */
template< class Matrix, class container1, class container2>
inline void gemv( Matrix& m, 
                  container1& x, 
                  container2& y)
{
    dg::blas2::detail::doGemv( m, x, y, 
                       typename dg::MatrixTraits<Matrix>::matrix_category(), 
                       typename dg::VectorTraits<container1>::vector_category(),
                       typename dg::VectorTraits<container2>::vector_category() );
    return;
}
/**
 * @brief General Matrix-container product
 *
 * @param alpha A Scalar
 * @param P The Matrix
 * @param x A container different from y 
 * @param beta A Scalar
 * @param y contains the solution on output
 */
template< class Precon, class container>
inline void gemv( typename MatrixTraits<Precon>::value_type alpha, 
                  const Precon& P, 
                  const container& x, 
                  typename MatrixTraits<Precon>::value_type beta, 
                  container& y)
{
    if(alpha == (typename MatrixTraits<Precon>::value_type)0) {
        dg::blas1::scal( y, alpha);
        return;
    }
    dg::blas2::detail::doGemv( alpha, P, x, beta, y, 
                       typename dg::MatrixTraits<Precon>::matrix_category(), 
                       typename dg::VectorTraits<container>::vector_category() );
    return;
}
///@}

} //namespace blas2
} //namespace dg

