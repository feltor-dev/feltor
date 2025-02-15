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
inline std::vector<int64_t> doDot_superacc( int* status, const ContainerType1& x, const MatrixType& m, const ContainerType2& y)
{
    static_assert(
            dg::is_vector_v<ContainerType1> && dg::is_vector_v<MatrixType> &&
            dg::is_vector_v<ContainerType2> ,
        "The container types must have a vector data layout (AnyVector)!");
    //check ContainerTypes: must be either scalar or same base category
    using vector_type = find_if_t<dg::is_not_scalar, ContainerType1, ContainerType1, ContainerType2>;
    using vector_category  = get_tensor_category<vector_type>;
    static_assert(
            dg::is_scalar_or_same_base_category<ContainerType1, vector_category>::value &&
            dg::is_scalar_or_same_base_category<ContainerType2, vector_category>::value,
        "All container types must be either Scalar or have compatible Vector categories (AnyVector or Same base class)!");
    return doDot_superacc( status, x, m, y, get_tensor_category<MatrixType>(), vector_category());
}

}//namespace detail
///@endcond

/*! @brief \f$ x^T M y\f$; Binary reproducible general dot product
 *
 * This routine computes the scalar product defined by the symmetric positive
 * definite matrix M \f[ x^T M y = \sum_{i,j=0}^{N-1} x_i M_{ij} y_j \f]
 *
 * For example
 * @snippet{trimleft} blas_t.cpp dot
 * Or a more elaborate use case
 * @snippet{trimleft} evaluation_t.cpp evaluate2d
 *
 * @attention if one of the input vectors contains \c Inf or \c NaN or the
 * product of the input numbers reaches \c Inf or \c Nan then the behaviour
 * is undefined and the function may throw. See @ref dg::ISNFINITE and @ref
 * dg::ISNSANE in that case
 * @note Our implementation guarantees **binary reproducible** results up to
 * and excluding the last mantissa bit of the result.  Furthermore, the sum is
 * computed with **infinite precision** and the result is then rounded to the
 * nearest double precision number. Although the products are not computed with
 * infinite precision, the order of multiplication is guaranteed.  This is
 * possible with the help of an adapted version of the \c dg::exblas library
 * and works for single and double precision.
 * @attention Binary Reproducible results are only guaranteed for **float** or
 * **double** input.  All other value types redirect to <tt>dg::blas1::vdot(
 * dg::Product(), x, m, y);</tt>
 *
 * @param x Left input
 * @param m The diagonal Matrix.
 * @param y Right input (may alias \c x)
 * @return Generalized scalar product. If \c x and \c y are vectors of
 * containers and \c m is not, then we sum the results of <tt>dg::blas2::dot(
 * x[i], m, y[i])</tt>
 * @note This routine is always executed synchronously due to the
    implicit memcpy of the result. With mpi the result is broadcasted to all
    processes. Also note that the behaviour is undefined when one of the
    containers contains \c nan
 * @tparam MatrixType \c MatrixType has to have a category derived from \c
 * AnyVectorTag and must be compatible with the \c ContainerTypes
 * @copydoc hide_ContainerType
 */
template< class ContainerType1, class MatrixType, class ContainerType2>
inline auto dot( const ContainerType1& x, const MatrixType& m, const ContainerType2& y)
{
    if constexpr (std::is_floating_point_v<get_value_type<ContainerType1>> &&
                  std::is_floating_point_v<get_value_type<MatrixType>>   &&
                  std::is_floating_point_v<get_value_type<ContainerType2>>)
    {
        int status = 0;
        std::vector<int64_t> acc = dg::blas2::detail::doDot_superacc( &status,
            x,m,y);
        if( status != 0)
            throw dg::Error(dg::Message(_ping_)<<"dg::blas2::dot failed "
                <<"since one of the inputs contains NaN or Inf");
        return exblas::cpu::Round(acc.data());
    }
    else
    {
        return dg::blas1::vdot( dg::Product(), x, m, y);
    }
}

/*! @brief \f$ x^T M x\f$; Binary reproducible general dot product
 *
 * Alias for \c dg::blas2::dot( x,m,x)
 * \f[ x^T M x = \sum_{i,j=0}^{N-1} x_i M_{ij} x_j \f]
 * @param m The diagonal Matrix
 * @param x Right input
 * @return Generalized scalar product. If \c x is a vector of containers and \c
 * m is not, then we sum the results of \c dg::blas2::dot( m, x[i])
 * @note This routine is always executed synchronously due to the
 *   implicit memcpy of the result.
 * @note This routine is equivalent to the call \c dg::blas2::dot( x, m, x);
 * which should be prefered because it looks more explicit
 * @tparam MatrixType \c MatrixType has to have a category derived from \c
 * AnyVectorTag and must be compatible with the \c ContainerTypes
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
    static_assert( std::is_same_v<get_execution_policy<ContainerType1>,
                                  get_execution_policy<ContainerType2>>,
                                "Vector types must have same execution policy");
    static_assert( std::is_same_v<get_value_type<ContainerType1>,
                                  get_value_type<MatrixType>> &&
                   std::is_same_v<get_value_type<ContainerType2>,
                                  get_value_type<MatrixType>>,
                                "Vector and Matrix types must have same value type");
    static_assert( std::is_same_v<get_tensor_category<ContainerType1>,
                                  get_tensor_category<ContainerType2>>,
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
    static_assert( std::is_same_v<get_execution_policy<ContainerType1>,
                                  get_execution_policy<ContainerType2>>,
                                "Vector types must have same execution policy");
    static_assert( std::is_same_v<get_value_type<ContainerType1>,
                                  get_value_type<MatrixType>> &&
                   std::is_same_v<get_value_type<ContainerType2>,
                                  get_value_type<MatrixType>>,
                                "Vector and Matrix types must have same value type");
    static_assert( std::is_same_v<get_tensor_category<ContainerType1>,
                                  get_tensor_category<ContainerType2>>,
                                "Vector types must have same data layout");
    dg::blas2::detail::doSymv( std::forward<MatrixType>(M), x, y,
            get_tensor_category<MatrixType>(),
            get_tensor_category<ContainerType1>());
}
template< class FunctorType, class MatrixType, class ContainerType1, class ContainerType2>
inline void doStencil(
                  FunctorType f,
                  MatrixType&& M,
                  const ContainerType1& x,
                  ContainerType2& y,
                  AnyMatrixTag)
{
    static_assert( std::is_same_v<get_execution_policy<ContainerType1>,
                                  get_execution_policy<ContainerType2>>,
                                "Vector types must have same execution policy");
    static_assert( std::is_same_v<get_value_type<ContainerType1>,
                                  get_value_type<MatrixType>> &&
                   std::is_same_v<get_value_type<ContainerType2>,
                                  get_value_type<MatrixType>>,
                                "Vector and Matrix types must have same value type");
    static_assert( std::is_same_v<get_tensor_category<ContainerType1>,
                                  get_tensor_category<ContainerType2>>,
                                "Vector types must have same data layout");
    dg::blas2::detail::doStencil( f, std::forward<MatrixType>(M), x, y,
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
 * @note There is nothing that prevents \c M from being non-symmetric or even
 * non-linear. In this sense the term "symv" (symmetrix-Matrix-Vector
 * multiplication) is misleading.  For better code readability we introduce
 * aliases: \c dg::blas2::gemv (general Matrix-Vector multiplication) and
 * \c dg::apply (general, possibly non-linear functor application).
 *
 * For example
 * @snippet{trimleft,cpp} operator_t.cpp symv 2
 * or a more elaborate use case
 * @snippet{trimleft} derivatives_t.cpp derive
 *
 * @param alpha A Scalar
 * @param M The Matrix.
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
 * @note There is nothing that prevents \c M from being non-symmetric or even
 * non-linear. In this sense the term "symv" (symmetrix-Matrix-Vector
 * multiplication) is misleading.  For better code readability we introduce
 * aliases: \c dg::blas2::gemv (general Matrix-Vector multiplication) and
 * \c dg::apply (general, possibly non-linear functor application)
 *
 * For example
 * @snippet{trimleft,cpp} operator_t.cpp symv 1
 * or a more elaborate use case
 * @snippet{trimleft} derivatives_t.cpp derive
 *
 * @param M The Matrix.
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
/*! @brief Alias for \c blas2::symv \f$ y = \alpha M x + \beta y \f$;
 *
 * This Alias exists for code readability: if your "symmetric matrix" is not
 * actually a symmetric matrix then it may seem unnatural to write \c
 * blas2::symv in your code
 * @ingroup blas2
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

/*! @brief Alias for \c blas2::symv \f$ y = M x\f$;
 *
 * This Alias exists for code readability: if your "symmetric matrix" is not
 * actually a symmetric matrix then it may seem unnatural to write \c
 * blas2::symv in your code
 * @ingroup blas2
 */
template< class MatrixType, class ContainerType1, class ContainerType2>
inline void gemv( MatrixType&& M,
                  const ContainerType1& x,
                  ContainerType2& y)
{
    dg::blas2::symv( std::forward<MatrixType>(M), x, y);
}

/**
 * @brief \f$ f(i, x_0, x_1, ...)\ \forall i\f$; Customizable and generic for loop
 *
 * @attention Only works for shared memory vectors (or scalars): no MPI, no Recursive (find reasons below).
 * @attention For trivially parallel operations (no neighboring points involved) use \c dg::blas1::subroutine
 *
 * This routine loops over an arbitrary user-defined "loop body" functor \c f with an arbitrary number of arguments \f$ x_s\f$ elementwise
 * \f[ f(i, x_{0}, x_{1}, ...)  \f]
 * where \c i iterates from \c 0 to a given size \c N.
 * The order of iterations is undefined.
 * It is equivalent to the following
 * @code
 * for(unsigned i=0; i<N; i++)
 *     f( i, &x_0[0], &x_1[0], ...);
 * @endcode
 * With this function very general for-loops can be parallelized like for
 * example
 * @snippet{trimleft} blas_t.cpp parallel_for
 * or
 * @snippet{trimleft} blas_t.cpp parallel_transpose
 *
 * @note In a way this function is a generalization of \c dg::blas1::subroutine
 * to non-trivial parallelization tasks. However, this comes at a price:
 * this function only works for containers with the \c dg::SharedVectorTag and sclar types.
 * The reason it cannot work for MPI is that the stencil (and thus the
 * communication pattern) is unkown. However, it can serve as an important
 * building block for other parallel functions like \c dg::blas2::stencil.
 * @note This is the closest function we have to <tt> kokkos::parallel_for</tt> of the <a href="https://github.com/kokkos/kokkos">Kokkos library</a>.
 *
 * @param f the loop body
 * @param N the total number of iterations in the for loop
 * @param x the first argument
 * @param xs other arguments
 * @attention The user has to decide whether or not it is safe to alias input or output vectors. If in doubt, do not alias output vectors.
 * @tparam Stencil a function or functor with an arbitrary number of arguments
 * and no return type; The first argument is an unsigned (the loop iteration),
 * afterwards takes a \c const_pointer_type argument (const pointer to first element in vector) for each input argument in
 * the call and a <tt> pointer_type  </tt> argument (pointer to first element in vector) for each output argument.
 * Scalars are forwarded "as is" <tt> scalar_type </tt>.
 * \c Stencil must be callable on the device in use. In particular, with CUDA
 * it must be a functor (@b not a function) and its signature must contain the
 * \__device__ specifier. (s.a. \ref DG_DEVICE)
  * @tparam ContainerType
  * Any class for which a specialization of \c TensorTraits exists and which
  * fulfills the requirements of the \c SharedVectorTag and \c AnyPolicyTag.
  * Among others
  *  - <tt> dg::HVec (serial), dg::DVec (cuda / omp)</tt>
  *  - \c int,  \c double and other primitive types ...
 */
template< class Stencil, class ContainerType, class ...ContainerTypes>
inline void parallel_for( Stencil f, unsigned N, ContainerType&& x, ContainerTypes&&... xs)
{
    // Is the assumption that results are automatically ready on return still true?
    // Do we have to introduce barriers around this function?
    static_assert( (dg::is_vector_v<ContainerType> &&
             ... && dg::is_vector_v<ContainerTypes>),
        "All container types must have a vector data layout (AnyVector)!");
    using vector_type = find_if_t<dg::is_not_scalar, ContainerType, ContainerType, ContainerTypes...>;
    using tensor_category  = get_tensor_category<vector_type>;
    static_assert(( dg::is_scalar_or_same_base_category<ContainerType, tensor_category>::value &&
              ... &&dg::is_scalar_or_same_base_category<ContainerTypes, tensor_category>::value),
        "All container types must be either Scalar or have compatible Vector categories (AnyVector or Same base class)!");
    dg::blas2::detail::doParallelFor(tensor_category(), f, N, std::forward<ContainerType>(x), std::forward<ContainerTypes>(xs)...);
}
/*! @brief \f$ F(M, x, y)\f$
 *
 * This routine calls \f[ F(i, [M], x, y) \f] for all \f$ i \in [0,N[\f$, where N is the number of rows in M,
 * using \c dg::blas2::parallel_for,
 * where [M] depends on the matrix type:
 *  - for a csr matrix it is [M] = m.row_offsets, m.column_indices, m.values
 * .
 * Possible shared memory implementation
 * @code
 * dg::blas2::parallel_for( F, m.num_rows, m.row_offsets, m.column_indices, m.values, x, y);
 * @endcode
 * Other matrix types have not yet been implemented.
 * @note Since the matrix is known, a communication pattern is available and thus the function works in parallel for MPI (unlike \c dg::blas2::parallel_for).
 * @note In a way this function is a generalization of \c dg::blas2::parallel_for to MPI vectors at the cost of having to encode the communication stencil in the matrix \c M and only one vector argument
 *
 * @param f The filter function is called like <tt> f(i, m.row_offsets_ptr, m.column_indices_ptr, m.values_ptr, x_ptr, y_ptr) </tt>
 * @param M The Matrix.
 * @param x input vector
 * @param y contains the solution on output (may not alias \p x)
 * @tparam FunctorType A type that is callable
 *  <tt> void operator()( unsigned, pointer, [m_pointers], const_pointer) </tt>  For GPU vector the functor
 *  must be callable on the device.
 * @tparam MatrixType So far only one of the \c cusp::csr_matrix types and their MPI variants <tt> dg::MPIDistMat<cusp::csr_matrix, Comm> </tt> are allowed
 * @sa dg::CSRMedianFilter, dg::create::window_stencil
 * @copydoc hide_ContainerType
 */
template< class FunctorType, class MatrixType, class ContainerType1, class ContainerType2>
inline void stencil(
                  FunctorType f,
                  MatrixType&& M,
                  const ContainerType1& x,
                  ContainerType2& y)
{
    dg::blas2::detail::doStencil( f, std::forward<MatrixType>(M), x, y, get_tensor_category<MatrixType>());
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
/*! @brief Alias for \c dg::blas2::symv \f$ y = \alpha M(x) + \beta y \f$;
 *
 * This Alias exists for code readability: if your matrix is not actually a matrix but
 * a functor then it may seem unnatural to write \c blas2::symv in your code especially
 * if  \c M is non-linear.
 * @ingroup blas2
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

/*! @brief Alias for \c dg::blas2::symv \f$ y = M( x)\f$;
 *
 * This Alias exists for code readability: if your matrix is not actually a matrix but
 * a functor then it may seem unnatural to write \c blas2::symv in your code especially
 * if  \c M is non-linear.
 * @ingroup blas2
 */
template< class MatrixType, class ContainerType1, class ContainerType2>
inline void apply( MatrixType&& M,
                  const ContainerType1& x,
                  ContainerType2& y)
{
    dg::blas2::symv( std::forward<MatrixType>(M), x, y);
}
} //namespace dg
