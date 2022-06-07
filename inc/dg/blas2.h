#pragma once

#include "backend/tensor_traits.h"
#include "backend/tensor_traits_std.h"
#include "backend/tensor_traits_thrust.h"
#include "backend/tensor_traits_cusp.h"
#include "backend/blas2_dispatch_scalar.h"
#include "backend/blas2_dispatch_shared.h"
#include "backend/blas2_cusp.h"
#include "backend/blas2_sparseblockmat.h"
#include "backend/blas2_selfmade.h"
#include "backend/blas2_densematrix.h"
#ifdef MPI_VERSION
#include "backend/blas2_dispatch_mpi.h"
#endif //MPI_VERSION
#include "backend/blas2_dispatch_vector.h"


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

///@cond
namespace detail{

template< class ContainerType1, class MatrixType, class ContainerType2>
inline std::vector<int64_t> doDot_superacc( const ContainerType1& x, const MatrixType& m, const ContainerType2& y)
{
    static_assert( all_true<
            dg::is_vector<ContainerType1>::value,
            dg::is_vector<MatrixType>::value,
            dg::is_vector<ContainerType2>::value>::value,
        "The container types must have a vector data layout (AnyVector)!");
    //check ContainerTypes: must be either scalar or same base category
    using vector_type = find_if_t<dg::is_not_scalar, ContainerType1, ContainerType1, ContainerType2>;
    using vector_category  = get_tensor_category<vector_type>;
    static_assert( all_true<
            dg::is_scalar_or_same_base_category<ContainerType1, vector_category>::value,
            dg::is_scalar_or_same_base_category<ContainerType2, vector_category>::value
            >::value,
        "All container types must be either Scalar or have compatible Vector categories (AnyVector or Same base class)!");
    return doDot_superacc( x, m, y, get_tensor_category<MatrixType>(), vector_category());
}

}//namespace detail
///@endcond

/*! @brief \f$ x^T M y\f$; Binary reproducible general dot product
 *
 * This routine computes the scalar product defined by the symmetric positive definite
 * matrix M \f[ x^T M y = \sum_{i,j=0}^{N-1} x_i M_{ij} y_j \f]
 *
 * @copydoc hide_code_evaluate2d
 * @attention if one of the input vectors contains \c Inf or \c NaN or the
 * product of the input numbers reaches \c Inf or \c Nan then the behaviour
 * is undefined and the function may throw. See @ref dg::ISNFINITE and @ref
 * dg::ISNSANE in that case
 * @note Our implementation guarantees **binary reproducible** results up to and excluding the last mantissa bit of the result.
 * Furthermore, the sum is computed with **infinite precision** and the result is then rounded
 * to the nearest double precision number. Although the products are not computed with
 * infinite precision, the order of multiplication is guaranteed.
 * This is possible with the help of an adapted version of the \c dg::exblas library and
* works for single and double precision.
 *
 * @param x Left input
 * @param m The diagonal Matrix.
 * @param y Right input (may alias \c x)
 * @return Generalized scalar product. If \c x and \c y are vectors of containers and \c m is not, then we sum the results of \c dg::blas2::dot( x[i], m, y[i])
 * @note This routine is always executed synchronously due to the
    implicit memcpy of the result. With mpi the result is broadcasted to all processes. Also note that the behaviour is undefined when one of the containers contains \c nan
 * @tparam MatrixType \c MatrixType has to have a category derived from \c AnyVectorTag and must be compatible with the \c ContainerTypes
 * @copydoc hide_ContainerType
 */
template< class ContainerType1, class MatrixType, class ContainerType2>
inline get_value_type<MatrixType> dot( const ContainerType1& x, const MatrixType& m, const ContainerType2& y)
{
    std::vector<int64_t> acc = dg::blas2::detail::doDot_superacc( x,m,y);
    return exblas::cpu::Round(acc.data());
}

/*! @brief \f$ x^T M x\f$; Binary reproducible general dot product
 *
 * Equivalent to \c dg::blas2::dot( x,m,x)
 * \f[ x^T M x = \sum_{i,j=0}^{N-1} x_i M_{ij} x_j \f]
 * @param m The diagonal Matrix
 * @param x Right input
 * @return Generalized scalar product. If \c x is a vector of containers and \c m is not, then we sum the results of \c dg::blas2::dot( m, x[i])
 * @note This routine is always executed synchronously due to the
    implicit memcpy of the result.
 * @note This routine is equivalent to the call \c dg::blas2::dot( x, m, x);
     which should be prefered because it looks more explicit
 * @tparam MatrixType \c MatrixType has to have a category derived from \c AnyVectorTag and must be compatible with the \c ContainerTypes
 * @copydoc hide_ContainerType
 */
template< class MatrixType, class ContainerType>
inline get_value_type<MatrixType> dot( const MatrixType& m, const ContainerType& x)
{
    return dg::blas2::dot( x, m, x);
}
///@cond
namespace detail{
//resolve tags in two stages: first the matrix and then the container type
template< class MatrixType, class ContainerType1, class ContainerType2>
inline void doSymv( get_value_type<ContainerType1> alpha,
                  MatrixType&& M,
                  const ContainerType1& x,
                  get_value_type<ContainerType1> beta,
                  ContainerType2& y,
                  AnyScalarTag)
{
    dg::blas1::pointwiseDot( alpha, std::forward<MatrixType>(M), x, beta, y);
}
template< class MatrixType, class ContainerType1, class ContainerType2>
inline void doSymv( MatrixType&& M,
                  const ContainerType1& x,
                  ContainerType2& y,
                  AnyScalarTag)
{
    dg::blas1::pointwiseDot( std::forward<MatrixType>(M), x, y);
}

template< class MatrixType, class ContainerType1, class ContainerType2>
inline void doSymv( get_value_type<ContainerType1> alpha,
                  MatrixType&& M,
                  const ContainerType1& x,
                  get_value_type<ContainerType1> beta,
                  ContainerType2& y,
                  AnyMatrixTag)
{
    static_assert( std::is_same<get_execution_policy<ContainerType1>,
                                get_execution_policy<ContainerType2>>::value,
                                "Vector types must have same execution policy");
    static_assert( std::is_same<get_value_type<ContainerType1>,
                                get_value_type<MatrixType>>::value &&
                   std::is_same<get_value_type<ContainerType2>,
                                get_value_type<MatrixType>>::value,
                                "Vector and Matrix types must have same value type");
    static_assert( std::is_same<get_tensor_category<ContainerType1>,
                                get_tensor_category<ContainerType2>>::value,
                                "Vector types must have same data layout");
    dg::blas2::detail::doSymv( alpha, std::forward<MatrixType>(M), x, beta, y,
            get_tensor_category<MatrixType>(),
            get_tensor_category<ContainerType1>());
}
template< class MatrixType, class ContainerType1, class ContainerType2>
inline void doSymv( MatrixType&& M,
                  const ContainerType1& x,
                  ContainerType2& y,
                  AnyMatrixTag)
{
    static_assert( std::is_same<get_execution_policy<ContainerType1>,
                                get_execution_policy<ContainerType2>>::value,
                                "Vector types must have same execution policy");
    static_assert( std::is_same<get_value_type<ContainerType1>,
                                get_value_type<MatrixType>>::value &&
                   std::is_same<get_value_type<ContainerType2>,
                                get_value_type<MatrixType>>::value,
                                "Vector and Matrix types must have same value type");
    static_assert( std::is_same<get_tensor_category<ContainerType1>,
                                get_tensor_category<ContainerType2>>::value,
                                "Vector types must have same data layout");
    dg::blas2::detail::doSymv( std::forward<MatrixType>(M), x, y,
            get_tensor_category<MatrixType>(),
            get_tensor_category<ContainerType1>());
}
template< class FunctorType, class MatrixType, class ContainerType1, class ContainerType2>
inline void doFilteredSymv( get_value_type<ContainerType1> alpha,
                  FunctorType f,
                  MatrixType&& M,
                  const ContainerType1& x,
                  get_value_type<ContainerType1> beta,
                  ContainerType2& y,
                  AnyMatrixTag)
{
    static_assert( std::is_same<get_execution_policy<ContainerType1>,
                                get_execution_policy<ContainerType2>>::value,
                                "Vector types must have same execution policy");
    static_assert( std::is_same<get_value_type<ContainerType1>,
                                get_value_type<MatrixType>>::value &&
                   std::is_same<get_value_type<ContainerType2>,
                                get_value_type<MatrixType>>::value,
                                "Vector and Matrix types must have same value type");
    static_assert( std::is_same<get_tensor_category<ContainerType1>,
                                get_tensor_category<ContainerType2>>::value,
                                "Vector types must have same data layout");
    dg::blas2::detail::doFilteredSymv( alpha, f, std::forward<MatrixType>(M), x, beta, y,
            get_tensor_category<MatrixType>(),
            get_tensor_category<ContainerType1>());
}

template< class MatrixType, class ContainerType1, class ContainerType2>
inline void doSymv( get_value_type<ContainerType1> alpha,
                  MatrixType&& M,
                  const ContainerType1& x,
                  get_value_type<ContainerType1> beta,
                  ContainerType2& y,
                  NotATensorTag)
{
    // if a compiler error message brings you here then you need to overload
    // the parenthesis operator for your class
    // void operator()( value_type alpha, const ContainerType1& x, value_type beta, ContainerType2& y)
    M(alpha,x,beta,y);
}
template< class MatrixType, class ContainerType1, class ContainerType2>
inline void doSymv( MatrixType&& M,
                  const ContainerType1& x,
                  ContainerType2& y,
                  NotATensorTag)
{
    // if a compiler error message brings you here then you need to overload
    // the parenthesis operator for your class
    // void operator()( const ContainerType1& x, ContainerType2& y)
    M(x,y);
}

}//namespace detail
///@endcond

/*! @brief \f$ y = \alpha M x + \beta y\f$
 *
 * This routine computes \f[ y = \alpha M x + \beta y \f]
 * where \f$ M\f$ is a matrix (or functor that is called like \c M(alpha,x,beta,y)).
 * @copydoc hide_code_blas2_symv
 * @param alpha A Scalar
 * @param M The Matrix.
 * There is nothing that prevents you from making the matrix \c M non-symmetric or even
 * non-linear. In this sense the term "symv" (symmetrix-Matrix-Vector
 * multiplication) is misleading.  For better code readability we introduce
 * aliases: \c dg::blas2::gemv (general Matrix-Vector multiplication) and
 * \c dg::apply (general, possibly non-linear functor application).
 * @param x input vector
 * @param beta A Scalar
 * @param y contains the solution on output (may not alias \p x)
 * @attention \p y may not alias \p x, the only exception is if \c MatrixType has the \c AnyVectorTag
 * @attention If y on input contains a NaN or Inf, it may contain NaN or Inf on
 * output as well even if beta is zero.
 * @copydoc hide_matrix
 * @copydoc hide_ContainerType
 */
template< class MatrixType, class ContainerType1, class ContainerType2>
inline void symv( get_value_type<ContainerType1> alpha,
                  MatrixType&& M,
                  const ContainerType1& x,
                  get_value_type<ContainerType1> beta,
                  ContainerType2& y)
{
    if(alpha == (get_value_type<ContainerType1>)0) {
        dg::blas1::scal( y, beta);
        return;
    }
    dg::blas2::detail::doSymv( alpha, std::forward<MatrixType>(M), x, beta, y, get_tensor_category<MatrixType>());
}



/*! @brief \f$ y = M x\f$
 *
 * This routine computes \f[ y = M x \f]
 * where \f$ M\f$ is a matrix (or functor that is called like \c M(x,y)).
 * @copydoc hide_code_blas2_symv
 * @param M The Matrix.
 * There is nothing that prevents you from making the matrix \c M non-symmetric or even
 * non-linear. In this sense the term "symv" (symmetrix-Matrix-Vector
 * multiplication) is misleading.  For better code readability we introduce
 * aliases: \c dg::blas2::gemv (general Matrix-Vector multiplication) and
 * \c dg::apply (general, possibly non-linear functor application)
 * @param x input vector
 * @param y contains the solution on output (may not alias \p x)
 * @attention \p y may not alias \p x, the only exception is if \c MatrixType has the \c AnyVectorTag and \c ContainerType1 ==\c ContainerType2
 * @attention If y on input contains a NaN or Inf it is not a prioriy clear
 * if it will contain a NaN or Inf on output as well.
 * Our own matrix formats overwrite y and work correctly but for third-party
 * libraries it is worth double-checking.
 * @copydoc hide_matrix
 * @copydoc hide_ContainerType
 */
template< class MatrixType, class ContainerType1, class ContainerType2>
inline void symv( MatrixType&& M,
                  const ContainerType1& x,
                  ContainerType2& y)
{
    dg::blas2::detail::doSymv( std::forward<MatrixType>(M), x, y, get_tensor_category<MatrixType>());
}
/*! @brief \f$ y = \alpha M x + \beta y \f$;
 * (alias for symv)
 *
 * @copydetails symv(get_value_type<ContainerType1>,MatrixType&&,const ContainerType1&,get_value_type<ContainerType1>,ContainerType2&)
 */
template< class MatrixType, class ContainerType1, class ContainerType2>
inline void gemv( get_value_type<ContainerType1> alpha,
                  MatrixType&& M,
                  const ContainerType1& x,
                  get_value_type<ContainerType1> beta,
                  ContainerType2& y)
{
    dg::blas2::symv( alpha, std::forward<MatrixType>(M), x, beta, y);
}

/*! @brief \f$ y = M x\f$;
 * (alias for symv)
 *
 * @copydetails symv(MatrixType&&,const ContainerType1&,ContainerType2&)
 */
template< class MatrixType, class ContainerType1, class ContainerType2>
inline void gemv( MatrixType&& M,
                  const ContainerType1& x,
                  ContainerType2& y)
{
    dg::blas2::symv( std::forward<MatrixType>(M), x, y);
}
/*! @brief \f$ y = \alpha F(M, x) + \beta y\f$
 *
 * This routine computes \f[ y_i = \alpha F(m^i, x^i, N^i) + \beta y_i \f] for every row \c i.
 * The matrix \f$ M\f$ is a sparse matrix that provides a stencil that is for each row it gathers
 * \f$ N^i\f$ elements of \f$ x\f$ and \f$ M\f$ into temporary vectors \f$ m^i,\ x^i\f$.
 * Then for each row the functor \f$ F \f$ is called on pointers to the first elements
 * and the number of elements in the two vectors. The number of elements \f$N^i\f$ in
 * \f$ m^i,\ x^i\f$ can vary in each row.
 * @note If F computes the dot product of its argument,
 * then the result is the matrix-vector product between M and x and \c dg::blas2::filtered_symv computes
 * the same as \c dg::blas2::symv
 *
 * @copydoc hide_code_blas2_filtered_symv
 * @param alpha A Scalar
 * @param f The filter function is called like \c result=f( m_ptr, x_ptr, size)
 * @param M The Matrix.
 * @param x input vector
 * @param beta A Scalar
 * @param y contains the solution on output (may not alias \p x)
 * @attention \p y may not alias \p x, the only exception is if \c MatrixType has the \c AnyVectorTag
 * @attention If y on input contains a NaN or Inf, it may contain NaN or Inf on
 * output as well even if beta is zero.
 * @tparam FunctorType A type that is callable
 *  <tt> result_type operator()( const_pointer, const_pointer, size_t) </tt>  where the first two arguments
 *  point to two contiguous arrays of the size given in the last argument. For GPU vector the functor
 *  must be callable on the device.
 * @copydoc hide_matrix
 * @attention So far only the \c cusp::ell_matrix type and its MPI variant is allowed
 * @copydoc hide_ContainerType
 */
template< class FunctorType, class MatrixType, class ContainerType1, class ContainerType2>
inline void filtered_symv( get_value_type<ContainerType1> alpha,
                  FunctorType f,
                  MatrixType&& M,
                  const ContainerType1& x,
                  get_value_type<ContainerType1> beta,
                  ContainerType2& y)
{
    if(alpha == (get_value_type<ContainerType1>)0) {
        dg::blas1::scal( y, beta);
        return;
    }
    dg::blas2::detail::doFilteredSymv( alpha, f, std::forward<MatrixType>(M), x, beta, y, get_tensor_category<MatrixType>());
}
/**
 * @brief \f$ y = x\f$; Generic way to copy and/or convert a Matrix type to a different Matrix type
 *
 * e.g. from CPU to GPU, or double to float, etc.
 * @copydoc hide_matrix
 * @tparam AnotherMatrix Another Matrix type
 * @param x source
 * @param y sink
 * @note y gets resized properly
 * @copydoc hide_code_blas2_symv
 */
template<class MatrixType, class AnotherMatrixType>
inline void transfer( const MatrixType& x, AnotherMatrixType& y)
{
    dg::blas2::detail::doTransfer( x,y,
            get_tensor_category<MatrixType>(),
            get_tensor_category<AnotherMatrixType>());
}
///@}

} //namespace blas2
/*! @brief \f$ y = \alpha M(x) + \beta y \f$;
 * (alias for \c dg::blas2::symv)
 *
 * This Alias exists for code readability: if your matrix is not actually a matrix but
 * a functor then it may seem unnatural to write \c blas2::symv in your code especially
 * if  \c M is non-linear.
 * @ingroup backend
 */
template< class MatrixType, class ContainerType1, class ContainerType2>
inline void apply( get_value_type<ContainerType1> alpha,
                  MatrixType&& M,
                  const ContainerType1& x,
                  get_value_type<ContainerType1> beta,
                  ContainerType2& y)
{
    dg::blas2::symv( alpha, std::forward<MatrixType>(M), x, beta, y);
}

/*! @brief \f$ y = M( x)\f$;
 * (alias for \c dg::blas2::symv)
 *
 * This Alias exists for code readability: if your matrix is not actually a matrix but
 * a functor then it may seem unnatural to write \c blas2::symv in your code especially
 * if  \c M is non-linear.
 * @ingroup backend
 */
template< class MatrixType, class ContainerType1, class ContainerType2>
inline void apply( MatrixType&& M,
                  const ContainerType1& x,
                  ContainerType2& y)
{
    dg::blas2::symv( std::forward<MatrixType>(M), x, y);
}
} //namespace dg
