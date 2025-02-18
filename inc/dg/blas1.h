#pragma once

#include "backend/predicate.h"
#include "backend/tensor_traits.h"
#include "backend/tensor_traits_scalar.h"
#include "backend/tensor_traits_thrust.h"
#include "backend/tensor_traits_cusp.h"
#include "backend/tensor_traits_std.h"
#include "backend/blas1_dispatch_scalar.h"
#include "backend/blas1_dispatch_shared.h"
#include "backend/tensor_traits_cusp.h"
#ifdef MPI_VERSION
#include "backend/mpi_vector.h"
#include "backend/blas1_dispatch_mpi.h"
#endif
#include "backend/blas1_dispatch_vector.h"
#include "backend/blas1_dispatch_map.h"
#include "subroutines.h"

/*!@file
 *
 * Basic linear algebra level 1 functions (functions that only involve vectors and not matrices)
 */

namespace dg{

/*! @brief BLAS Level 1 routines
 *
 * @ingroup blas1
 *
 * @note successive calls to blas routines are executed sequentially
 * @note A manual synchronization of threads or devices is never needed in an application
 * using these functions. All functions returning a value block until the value is ready.
 */
namespace blas1
{
///@cond
template< class ContainerType, class BinarySubroutine, class Functor, class ContainerType0, class ...ContainerTypes>
inline void evaluate( ContainerType& y, BinarySubroutine f, Functor g, const ContainerType0& x0, const ContainerTypes& ...xs);
///@endcond

///@addtogroup blas1
///@{

/**
 * @class hide_iterations
 * where \c i iterates over @b all elements inside the given vectors. The order of iterations is undefined. Scalar arguments to container types are interpreted as vectors with all elements constant. If \c ContainerType has the \c RecursiveVectorTag, \c i recursively loops over all entries.
 * If the vector sizes do not match, the result is undefined.
 * The compiler chooses the implementation and parallelization of this function based on given template parameters. For a full set of rules please refer to \ref dispatch.
 */
/**
 * @class hide_naninf
 * @attention only result vectors that are **write-only** and do not alias
 * input vectors contain correct results when the result vector contains NaN
 * or Inf on input. In particular, \c dg::blas1::scal( y, 0 ) does not remove
 * NaN or Inf from y while \c dg::blas1::copy( 0, y ) does.
 */

/*! @brief \f$ \sum_i f(x_{0i}, x_{1i}, ...)\f$ Extended Precision transform reduce
 *
 * This routine computes \f[ \sum_i f(x_{0i}, x_{1i}, ...)\f]
 * @copydoc hide_iterations
 *
 * For example
 * @snippet{trimleft} blas1_t.cpp vdot
 * or
 * @snippet{trimleft} blas1_t.cpp vcdot
 * @note The main motivator for this version of \c dot is that it works for complex numbers.
 * @attention if one of the input vectors contains \c Inf or \c NaN or the
 * product of the input numbers reaches \c Inf or \c Nan then the behaviour
 * is undefined and the function may throw. See @ref dg::ISNFINITE and @ref
 * dg::ISNSANE in that case
 * @note This implementation does **not guarantee binary reproducible** results.
 * The sum is computed with **extended precision** and the result is rounded
 * to the nearest double precision number.
 * This is possible with the help of an adapted version of the \c dg::exblas library and
 * works for single and double precision.

 * @tparam Functor signature: <tt> value_type_g operator()( value_type_x0, value_type_x1, ...) </tt>
 * @attention \c Functor must be callable on the device in use. In particular,
 * with CUDA it must be a functor tpye (@b not function) and its signatures
 * must contain the \__device__ specifier. (s.a. \ref DG_DEVICE)
 * @param f The functor to evaluate, see @ref functions and @ref variadic_evaluates for a collection of predefined functors to use here
 * @param x First input
 * @param xs More input (may alias x)
 * @return Scalar product as defined above
 * @note This routine is always executed synchronously due to the
        implicit memcpy of the result. With mpi the result is broadcasted to all processes.
 * @copydoc hide_ContainerType
 */
template<class Functor, class ContainerType, class ...ContainerTypes>
auto vdot( Functor f, const ContainerType& x, const ContainerTypes& ...xs) ->
    std::invoke_result_t<Functor, dg::get_value_type<ContainerType>, dg::get_value_type<ContainerTypes>...>
{
    // The reason it is called vdot and not dot is because of the amgiuity when vdot is called
    // with two arguments
    using T = std::invoke_result_t<Functor, dg::get_value_type<ContainerType>, dg::get_value_type<ContainerTypes>...>;

    int status = 0;
    if constexpr( std::is_integral_v<T>) // e.g. T = int
    {
        std::array<T, 1> fpe;
        dg::blas1::detail::doDot_fpe( &status, fpe, f, x, xs ...);
        if( fpe[0] - fpe[0] != T(0))
            throw dg::Error(dg::Message(_ping_)
                <<"dg::blas1::vdot (integral type) failed "
                <<"since one of the inputs contains NaN or Inf");
        return fpe[0];
    }
    else
    {
        constexpr size_t N = 3;
        std::array<T, N> fpe;
        dg::blas1::detail::doDot_fpe( &status, fpe, f, x, xs ...);
        for( unsigned u=0; u<N; u++)
        {
            if( fpe[u] - fpe[u] != T(0))
                throw dg::Error(dg::Message(_ping_)
                    <<"dg::blas1::vdot (floating type) failed "
                    <<"since one of the inputs contains NaN or Inf");
        }
        return exblas::cpu::Round(fpe);
    }
}

/*! @brief \f$ x^T y\f$ Binary reproducible Euclidean dot product between two vectors
 *
 * This routine computes \f[ x^T y = \sum_{i=0}^{N-1} x_i y_i \f]
 * @copydoc hide_iterations
 *
 * For example
 * @snippet{trimleft} blas1_t.cpp dot
 * or
 * @snippet{trimleft} blas1_t.cpp cdot
 * @attention if one of the input vectors contains \c Inf or \c NaN or the
 * product of the input numbers reaches \c Inf or \c Nan then the behaviour
 * is undefined and the function may throw. See @ref dg::ISNFINITE and @ref
 * dg::ISNSANE in that case
 * @note Our implementation guarantees **binary reproducible** results.
 * The sum is computed with **infinite precision** and the result is rounded
 * to the nearest double precision number.
 * This is possible with the help of an adapted version of the \c dg::exblas library and
* works for single and double precision.
* @attention Binary Reproducible results are only guaranteed for **float** or **double** input.
* All other value types redirect to <tt> dg::blas1::vdot( dg::Product(), x, y);</tt>

 * @param x Left Container
 * @param y Right Container may alias x
 * @return Scalar product as defined above
 * @note This routine is always executed synchronously due to the
        implicit memcpy of the result. With mpi the result is broadcasted to all processes.
 * @copydoc hide_ContainerType
 */
template< class ContainerType1, class ContainerType2>
inline auto dot( const ContainerType1& x, const ContainerType2& y)
{
    if constexpr (std::is_floating_point_v<get_value_type<ContainerType1>> &&
                  std::is_floating_point_v<get_value_type<ContainerType2>>)
    {
        int status = 0;
        std::vector<int64_t> acc = dg::blas1::detail::doDot_superacc( &status,
            x,y);
        if( status != 0)
            throw dg::Error(dg::Message(_ping_)<<"dg::blas1::dot failed "
                <<"since one of the inputs contains NaN or Inf");
        return exblas::cpu::Round(acc.data());
    }
    else
    {
        return dg::blas1::vdot( dg::Product(), x, y);
    }
}


/*! @brief \f$ f(x_0) \otimes f(x_1) \otimes \dots \otimes f(x_{N-1}) \f$ Custom (transform) reduction
 *
 * This routine computes \f[ s = f(x_0) \otimes f(x_1) \otimes \dots \otimes f(x_i) \otimes \dots \otimes f(x_{N-1}) \f]
 * where \f$ \otimes \f$ is an arbitrary **commutative** and **associative** binary operator, \f$ f\f$ is an optional unary operator and
 * @copydoc hide_iterations
 *
 * @note numerical addition/multiplication is **not** exactly associative
 * which means that the associated reduction looses precision due to inexact arithmetic. For binary reproducible exactly rounded results use the dg::blas1::dot function.
 * However, this function is more general and faster to execute than dg::blas1::dot.

 * For example
 * @snippet{trimleft} blas1_t.cpp reduce nan
 * or
 * @snippet{trimleft} blas1_t.cpp reduce min
 * @param x Container to reduce
 * @param zero The neutral element with respect to binary_op that is
 * <tt> x == binary_op( zero, x) </tt>. Determines the \c OutputType so make
 * sure to make the type clear to the compiler (e.g. write <tt> (double)0 </tt> instead
 * of \c 0 if you want \c double output)
 * @attention In the current implementation \c zero is used to initialize
 * partial sums e.g. when reducing MPI Vectors so it is important that \c zero
 * is actually the neutral element. The reduction will yield wrong results
 * if it is not.
 * @param binary_op an associative and commutative binary operator
 * @param unary_op a unary operator applies to each element of \c x
 * @return Custom reduction as defined above
 * @note This routine is always executed synchronously due to the
        implicit memcpy of the result. With mpi the result is broadcasted to
        all processes
 * @tparam BinaryOp Functor with signature: <tt> value_type  operator()(
 * value_type, value_type) </tt>, must be associative and commutative.
 * \c value_tpye must be compatible with \c OutputType
 * @tparam UnaryOp a unary operator. The argument type must be compatible with
 * \c get_value_type<ContainerType>. The return type must be convertible to
 * \c OutputType
 * @tparam OutputType The type of the result. Infered from \c zero so make sure
 * \c zero's type is clear to the compiler.
 * @copydoc hide_ContainerType
 * @sa For partial reductions see \c dg::Average
 */
template< class ContainerType, class OutputType, class BinaryOp, class UnaryOp
    = IDENTITY>
inline OutputType reduce( const ContainerType& x, OutputType zero, BinaryOp
        binary_op, UnaryOp unary_op = UnaryOp())
{
    //init must indeed have the same type as the values of Container since op must be associative
    // The generalization would be a transform_reduce combining subroutine and reduce
    return dg::blas1::detail::doReduce(
            dg::get_tensor_category<ContainerType>(), x, zero, binary_op,
            unary_op);
}

/**
 * @brief \f$ y=x \f$
 *
 * explicit pointwise assignment \f$ y_i = x_i\f$
 * @copydoc hide_iterations
 *
 * For example
 * @snippet{trimleft} blas1_t.cpp copy
 * @param source vector to copy
 * @param target (write-only) destination
 * @note in contrast to the \c dg::assign functions the \c copy function uses
 * the execution policy to determine the implementation and thus works
 * only on types with same execution policy
 * @note catches self-assignment
 * @copydoc hide_naninf
 * @copydoc hide_ContainerType
 */
template<class ContainerTypeIn, class ContainerTypeOut>
inline void copy( const ContainerTypeIn& source, ContainerTypeOut& target){
    if( std::is_same_v<ContainerTypeIn, ContainerTypeOut> && &source==(const ContainerTypeIn*)&target)
        return;
    dg::blas1::subroutine( dg::equals(), source, target);
}

/*! @brief \f$ x = \alpha x\f$
 *
 * This routine computes \f[ \alpha x_i \f]
 * @copydoc hide_iterations
 *
 * For example
 * @snippet{trimleft} blas1_t.cpp scal
 * @param alpha Scalar
 * @param x (read/write) x
 * @copydoc hide_naninf
 * @copydoc hide_ContainerType
 * @copydoc hide_value_type
 */
template< class ContainerType, class value_type>
inline void scal( ContainerType& x, value_type alpha)
{
    if( alpha == value_type(1))
        return;
    dg::blas1::subroutine( dg::Scal<value_type>(alpha), x );
}

/*! @brief \f$ x = x + \alpha \f$
 *
 * This routine computes \f[ x_i + \alpha \f]
 * @copydoc hide_iterations
 *
 * For example
 * @snippet{trimleft} blas1_t.cpp plus
 * @param alpha Scalar
 * @param x (read/write) x
 * @copydoc hide_naninf
 * @copydoc hide_ContainerType
 * @copydoc hide_value_type
 */
template< class ContainerType, class value_type>
inline void plus( ContainerType& x, value_type alpha)
{
    if( alpha == value_type(0))
        return;
    dg::blas1::subroutine( dg::Plus(alpha), x );
}

/*! @brief \f$ y = \alpha x + \beta y\f$
 *
 * This routine computes \f[ y_i =  \alpha x_i + \beta y_i \f]
 * @copydoc hide_iterations
 *
 * For example
 * @snippet{trimleft} blas1_t.cpp axpby
 * @param alpha Scalar
 * @param x ContainerType x may alias y
 * @param beta Scalar
 * @param y (read/write) ContainerType y contains solution on output
 * @copydoc hide_naninf
 * @copydoc hide_ContainerType
 */
template< class ContainerType, class ContainerType1, class value_type, class value_type1>
inline void axpby( value_type alpha, const ContainerType1& x, value_type1 beta, ContainerType& y)
{
    if( alpha == value_type(0) ) {
        scal( y, beta);
        return;
    }
    if( std::is_same_v<ContainerType, ContainerType1> && &x==(const ContainerType1*)&y){
        dg::blas1::scal( y, (alpha+beta));
        return;
    }
    dg::blas1::subroutine( dg::Axpby(alpha, beta),  x, y);
}

/*! @brief \f$ z = \alpha x + \beta y + \gamma z\f$
 *
 * This routine computes \f[ z_i =  \alpha x_i + \beta y_i + \gamma z_i \f]
 * @copydoc hide_iterations
 *
 * For example
 * @snippet{trimleft} blas1_t.cpp axpby
 * @param alpha Scalar
 * @param x ContainerType x may alias result
 * @param beta Scalar
 * @param y ContainerType y may alias result
 * @param gamma Scalar
 * @param z (read/write) ContainerType contains solution on output
 * @copydoc hide_naninf
 * @copydoc hide_ContainerType
 * @copydoc hide_value_type
 */
template< class ContainerType, class ContainerType1, class ContainerType2, class value_type, class value_type1, class value_type2>
inline void axpbypgz( value_type alpha, const ContainerType1& x, value_type1 beta, const ContainerType2& y, value_type2 gamma, ContainerType& z)
{
    if( alpha == value_type(0) )
    {
        axpby( beta, y, gamma, z);
        return;
    }
    else if( beta == value_type1(0) )
    {
        axpby( alpha, x, gamma, z);
        return;
    }
    if( std::is_same_v<ContainerType1, ContainerType2> && &x==(const ContainerType1*)&y){
        dg::blas1::axpby( alpha+beta, x, gamma, z);
        return;
    }
    else if( std::is_same_v<ContainerType1, ContainerType> && &x==(const ContainerType1*)&z){
        dg::blas1::axpby( beta, y, alpha+gamma, z);
        return;
    }
    else if( std::is_same_v<ContainerType2, ContainerType> && &y==(const ContainerType2*)&z){
        dg::blas1::axpby( alpha, x, beta+gamma, z);
        return;
    }
    dg::blas1::subroutine( dg::Axpbypgz(alpha, beta, gamma),  x, y, z);
}

/*! @brief \f$ z = \alpha x + \beta y\f$
 *
 * This routine computes \f[ z_i =  \alpha x_i + \beta y_i \f]
 * @copydoc hide_iterations
 *
 * For example
 * @snippet{trimleft} blas1_t.cpp axpbyz
 *
 * @param alpha Scalar
 * @param x ContainerType x may alias z
 * @param beta Scalar
 * @param y ContainerType y may alias z
 * @param z (write-only) ContainerType z contains solution on output
 * @copydoc hide_naninf
 * @copydoc hide_ContainerType
 * @copydoc hide_value_type
 */
template< class ContainerType, class ContainerType1, class ContainerType2, class value_type, class value_type1>
inline void axpby( value_type alpha, const ContainerType1& x, value_type1 beta, const ContainerType2& y, ContainerType& z)
{
    dg::blas1::evaluate( z , dg::equals(), dg::PairSum(), alpha, x, beta, y);
}

/**
 * @brief \f$ y = \alpha x_1 x_2 + \beta y\f$
 *
 * Multiplies two vectors element by element: \f[ y_i = \alpha x_{1i}x_{2i} + \beta y_i\f]
 * @copydoc hide_iterations
 *
 * For example
 * @snippet{trimleft} blas1_t.cpp pointwiseDot
 *
 * @param alpha scalar
 * @param x1 ContainerType x1
 * @param x2 ContainerType x2 may alias x1
 * @param beta scalar
 * @param y (read/write)  ContainerType y contains result on output ( may alias x1 or x2)
 * @copydoc hide_naninf
 * @copydoc hide_ContainerType
 * @copydoc hide_value_type
 */
template< class ContainerType, class ContainerType1, class ContainerType2, class value_type, class value_type1>
inline void pointwiseDot( value_type alpha, const ContainerType1& x1, const ContainerType2& x2, value_type1 beta, ContainerType& y)
{
    if( alpha == value_type(0) ) {
        dg::blas1::scal(y, beta);
        return;
    }
    //not sure this is necessary performance-wise, subroutine does allow aliases
    if( std::is_same_v<ContainerType, ContainerType1> && &x1==(const ContainerType1*)&y){
        dg::blas1::subroutine( dg::AxyPby(alpha,beta), x2, y );

        return;
    }
    if( std::is_same_v<ContainerType, ContainerType2> && &x2==(const ContainerType2*)&y){
        dg::blas1::subroutine( dg::AxyPby(alpha,beta), x1, y );

        return;
    }
    dg::blas1::subroutine( dg::PointwiseDot(alpha,beta), x1, x2, y );
}

/*! @brief \f$ y = x_1 x_2 \f$
*
* Multiplies two vectors element by element: \f[ y_i = x_{1i}x_{2i}\f]
* @copydoc hide_iterations
*
* For example
* @snippet{trimleft} blas1_t.cpp pointwiseDot 2
*
* @param x1 ContainerType x1
* @param x2 ContainerType x2 may alias x1
* @param y (write-only) ContainerType y contains result on output ( may alias x1 or x2)
* @copydoc hide_naninf
* @copydoc hide_ContainerType
*/
template< class ContainerType, class ContainerType1, class ContainerType2>
inline void pointwiseDot( const ContainerType1& x1, const ContainerType2& x2, ContainerType& y)
{
    dg::blas1::evaluate( y, dg::equals(), dg::PairSum(), x1,x2);
}

/**
* @brief \f$ y = \alpha x_1 x_2 x_3 + \beta y\f$
*
* Multiplies three vectors element by element: \f[ y_i = \alpha x_{1i}x_{2i}x_{3i} + \beta y_i\f]
* @copydoc hide_iterations
*
* For example
* @snippet{trimleft} blas1_t.cpp pointwiseDot 3
*
* @param alpha scalar
* @param x1 ContainerType x1
* @param x2 ContainerType x2 may alias x1
* @param x3 ContainerType x3 may alias x1 and/or x2
* @param beta scalar
* @param y  (read/write) ContainerType y contains result on output ( may alias x1,x2 or x3)
* @copydoc hide_naninf
* @copydoc hide_ContainerType
* @copydoc hide_value_type
*/
template< class ContainerType, class ContainerType1, class ContainerType2, class ContainerType3, class value_type, class value_type1>
inline void pointwiseDot( value_type alpha, const ContainerType1& x1, const ContainerType2& x2, const ContainerType3& x3, value_type1 beta, ContainerType& y)
{
    if( alpha == value_type(0) ) {
        dg::blas1::scal(y, beta);
        return;
    }
    dg::blas1::subroutine( dg::PointwiseDot(alpha,beta), x1, x2, x3, y );
}

/**
* @brief \f$ y = \alpha x_1/ x_2 + \beta y \f$
*
* Divides two vectors element by element: \f[ y_i = \alpha x_{1i}/x_{2i} + \beta y_i \f]

* @copydoc hide_iterations
*
* For example
* @snippet{trimleft} blas1_t.cpp pointwiseDivide
*
* @param alpha scalar
* @param x1 ContainerType x1
* @param x2 ContainerType x2 may alias x1
* @param beta scalar
* @param y  (read/write) ContainerType y contains result on output ( may alias x1 and/or x2)
* @copydoc hide_naninf
* @copydoc hide_ContainerType
* @copydoc hide_value_type
*/
template< class ContainerType, class ContainerType1, class ContainerType2, class value_type, class value_type1>
inline void pointwiseDivide( value_type alpha, const ContainerType1& x1, const ContainerType2& x2, value_type1 beta, ContainerType& y)
{
    if( alpha == value_type(0) ) {
        dg::blas1::scal(y, beta);
        return;
    }
    if( std::is_same_v<ContainerType, ContainerType1> && &x1==(const ContainerType1*)&y){
        dg::blas1::subroutine( dg::PointwiseDivide(alpha,beta), x2, y );

        return;
    }
    dg::blas1::subroutine( dg::PointwiseDivide(alpha, beta), x1, x2, y );
}

/**
* @brief \f$ y = x_1/ x_2\f$
*
* Divides two vectors element by element: \f[ y_i = x_{1i}/x_{2i}\f]
* @copydoc hide_iterations
*
* For example
* @snippet{trimleft} blas1_t.cpp pointwiseDivide 2
*
* @param x1 ContainerType x1
* @param x2 ContainerType x2 may alias x1
* @param y  (write-only) ContainerType y contains result on output ( may alias x1 and/or x2)
* @copydoc hide_naninf
* @copydoc hide_ContainerType
*/
template< class ContainerType, class ContainerType1, class ContainerType2>
inline void pointwiseDivide( const ContainerType1& x1, const ContainerType2& x2, ContainerType& y)
{
    dg::blas1::evaluate( y, dg::equals(), dg::divides(), x1, x2);
}

/**
* @brief \f$ z = \alpha x_1y_1 + \beta x_2y_2 + \gamma z\f$
*
* Multiplies and adds vectors element by element: \f[ z_i = \alpha x_{1i}y_{1i} + \beta x_{2i}y_{2i} + \gamma z_i \f]
* @copydoc hide_iterations
*
* For example
* @snippet{trimleft} blas1_t.cpp pointwiseDot 4
*
* @param alpha scalar
* @param x1 ContainerType x1
* @param y1 ContainerType y1
* @param beta scalar
* @param x2 ContainerType x2
* @param y2 ContainerType y2
* @param gamma scalar
* @param z  (read/write) ContainerType z contains result on output
* @note all aliases are allowed
* @copydoc hide_naninf
* @copydoc hide_ContainerType
* @copydoc hide_value_type
*/
template<class ContainerType, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerType4, class value_type, class value_type1, class value_type2>
void pointwiseDot(  value_type alpha, const ContainerType1& x1, const ContainerType2& y1,
                    value_type1 beta,  const ContainerType3& x2, const ContainerType4& y2,
                    value_type2 gamma, ContainerType & z)
{
    if( alpha==value_type(0)){
        pointwiseDot( beta, x2,y2, gamma, z);
        return;
    }
    else if( beta==value_type1(0)){
        pointwiseDot( alpha, x1,y1, gamma, z);
        return;
    }
    dg::blas1::subroutine( dg::PointwiseDot2(alpha, beta, gamma), x1, y1, x2, y2, z );
}

/*! @brief \f$ y = op(x)\f$
 *
 * This routine computes \f[ y_i = op(x_i) \f]
 * @copydoc hide_iterations
 *
 * For example
 * @snippet{trimleft} blas1_t.cpp transform
 *
 * @param x ContainerType x may alias y
 * @param y (write-only) ContainerType y contains result, may alias x
 * @param op unary %SquareMatrix to use on every element
 * @tparam UnaryOp Functor with signature: <tt> value_type operator()( value_type) </tt>
 * @note \c UnaryOp must be callable on the device in use. In particular, with CUDA it must be of functor tpye (@b not a function) and its signatures must contain the \__device__ specifier. (s.a. \ref DG_DEVICE)
 * @copydoc hide_naninf
 * @copydoc hide_ContainerType
 */
template< class ContainerType, class ContainerType1, class UnaryOp>
inline void transform( const ContainerType1& x, ContainerType& y, UnaryOp op )
{
    dg::blas1::subroutine( dg::Evaluate<dg::equals, UnaryOp>(dg::equals(),op), y, x);
}

/*! @brief \f$ f(g(x_0,x_1,...), y)\f$
 *
 * This routine elementwise evaluates \f[ f(g(x_{0i}, x_{1i}, ...), y_i) \f]
 * @copydoc hide_iterations
 *
 * For example
 * @snippet{trimleft} blas1_t.cpp evaluate
 *
 * @tparam BinarySubroutine Functor with signature: <tt> void ( value_type_g, value_type_y&) </tt> i.e. it reads the first (and second) and writes into the second argument
 * @tparam Functor signature: <tt> value_type_g operator()( value_type_x0, value_type_x1, ...) </tt>
 * @attention Both \c BinarySubroutine and \c Functor must be callable on the device in use. In particular, with CUDA they must be functor tpyes (@b not functions) and their signatures must contain the \__device__ specifier. (s.a. \ref DG_DEVICE)
 * @param y contains result
 * @param f The subroutine, for example \c dg::equals or \c dg::plus_equals, see @ref binary_operators for a collection of predefined functors to use here
 * @param g The functor to evaluate, see @ref functions and @ref variadic_evaluates for a collection of predefined functors to use here
 * @param x0 first input
 * @param xs more input
 * @note all aliases allowed
 * @copydoc hide_naninf
 * @copydoc hide_ContainerType
 *
 */
template< class ContainerType, class BinarySubroutine, class Functor, class ContainerType0, class ...ContainerTypes>
inline void evaluate( ContainerType& y, BinarySubroutine f, Functor g, const ContainerType0& x0, const ContainerTypes& ...xs)
{
    dg::blas1::subroutine( dg::Evaluate<BinarySubroutine, Functor>(f,g), y, x0, xs...);
}


///@cond
namespace detail{

template< class T, size_t N, class Functor, class ContainerType, class ...ContainerTypes>
inline void doDot_fpe( int* status, std::array<T,N>& fpe, Functor f,
    const ContainerType& x, const ContainerTypes& ...xs)
{
    static_assert( ( dg::is_vector_v<ContainerType> && ...
                  && dg::is_vector_v<ContainerTypes>),
        "All container types must have a vector data layout (AnyVector)!");
    using vector_type = find_if_t<dg::is_not_scalar, ContainerType, ContainerType, ContainerTypes...>;
    using tensor_category  = get_tensor_category<vector_type>;
    static_assert( ( dg::is_scalar_or_same_base_category<ContainerType, tensor_category>::value &&
              ... && dg::is_scalar_or_same_base_category<ContainerTypes, tensor_category>::value),
        "All container types must be either Scalar or have compatible Vector categories (AnyVector or Same base class)!");
    return doDot_fpe( tensor_category(), status, fpe, f, x, xs ...);

}

template< class ContainerType1, class ContainerType2>
inline std::vector<int64_t> doDot_superacc( int * status, const ContainerType1& x, const ContainerType2& y)
{
    static_assert( ( dg::is_vector_v<ContainerType1> && dg::is_vector_v<ContainerType2>),
        "All container types must have a vector data layout (AnyVector)!");
    using vector_type = find_if_t<dg::is_not_scalar, ContainerType1, ContainerType1, ContainerType2>;
    using tensor_category  = get_tensor_category<vector_type>;
    static_assert( ( dg::is_scalar_or_same_base_category<ContainerType1, tensor_category>::value
                  && dg::is_scalar_or_same_base_category<ContainerType2, tensor_category>::value),
        "All container types must be either Scalar or have compatible Vector categories (AnyVector or Same base class)!");
    return doDot_superacc( status, x, y, tensor_category());
}

}//namespace detail
///@endcond

/**
 * @brief \f$ f(x_0, x_1, ...)\f$; Customizable and generic blas1 function
 *
 * This routine evaluates an arbitrary user-defined subroutine \c f with an arbitrary number of arguments \f$ x_s\f$ elementwise
 * \f[ f(x_{0i}, x_{1i}, ...)  \f]
 * @copydoc hide_iterations
 *
 * For example
 * @snippet{trimleft} blas1_t.cpp subroutine
 *
 * @param f the subroutine, see @ref variadic_subroutines for a collection of predefind subroutines to use here
 * @param x the first argument
 * @param xs other arguments
@note This function can compute @b any trivial parallel expression for @b any
number of input and output arguments, which is quite remarkable really. In this sense it replaces all other \c blas1 functions
except the scalar product, which is not trivially parallel.
@attention The user has to decide whether or not it is safe to alias input or output vectors. If in doubt, do not alias output vectors.
 * @tparam Subroutine a function or functor with an arbitrary number of arguments and no return type; taking a \c value_type argument for each input argument in the call
 * and a <tt> value_type&  </tt> argument for each output argument.
 * \c Subroutine must be callable on the device in use. In particular, with CUDA it must be a functor (@b not a function) and its signature must contain the \__device__ specifier. (s.a. \ref DG_DEVICE)
 * @copydoc hide_naninf
 * @copydoc hide_ContainerType
 */
template< class Subroutine, class ContainerType, class ...ContainerTypes>
inline void subroutine( Subroutine f, ContainerType&& x, ContainerTypes&&... xs)
{
    static_assert( ( dg::is_vector_v<ContainerType> &&
               ...&& dg::is_vector_v<ContainerTypes>),
        "All container types must have a vector data layout (AnyVector)!");
    using vector_type = find_if_t<dg::is_not_scalar, ContainerType, ContainerType, ContainerTypes...>;
    using tensor_category  = get_tensor_category<vector_type>;
    static_assert( ( dg::is_scalar_or_same_base_category<ContainerType, tensor_category>::value &&
              ... && dg::is_scalar_or_same_base_category<ContainerTypes, tensor_category>::value),
        "All container types must be either Scalar or have compatible Vector categories (AnyVector or Same base class)!");
    dg::blas1::detail::doSubroutine(tensor_category(), f, std::forward<ContainerType>(x), std::forward<ContainerTypes>(xs)...);
}

/*! @brief \f$ f(g(x_{0i_0},x_{1i_1},...), y_I)\f$ (Kronecker evaluation)
 *
 * This routine elementwise evaluates \f[ f(g(x_{0i_0}, x_{1i_1}, ..., x_{(n-1)i_{n-1}}), y_{((i_{n-1} N_{n-2} +...)N_1+i_1)N_0+i_0}) \f]
 * for all @b combinations of input values.
 * \f$ N_i\f$ is the size of the vector \f$ x_i\f$.
 * The **first index \f$i_0\f$ is the fastest varying in the output**, then \f$ i_1\f$, etc.
 * If \f$ x_i\f$ is a scalar then the size \f$ N_i = 1\f$.
 * @attention None of the \f$ x_i\f$ or \f$ y\f$ can have the \c dg::RecursiveVectorTag
 *
 * The size of the output \f$ y\f$ must match the product of sizes of input vectors i.e.
 * \f[ N_y = \prod_{i=0}^{n-1} N_i \f]
 * The order of evaluations is undefined.
 * The compiler chooses the implementation and parallelization of this function based on given template parameters. For a full set of rules please refer to \ref dispatch.
 *
 * For example
 * @snippet{trimleft} blas1_t.cpp kronecker
 *
 * @note This function is trivially parallel and the MPI version simply calls the appropriate shared memory version
 * The user is responsible for making sure that the result has the correct communicator
 * @note For the function \f$ f(x_0, x_1, ..., x_{n-1}) = x_0 x_1 ... x_{n-1} \f$ <tt> dg::blas1::kronecker(y, dg::equals(), x_0, x_1, ...) </tt>computes the actual Kronecker product of the arguments **in reversed order** \f[ y = x_{n-1} \otimes x_{n-2} \otimes ... \otimes x_1 \otimes x_0\f] (or the outer product)
 * With this behaviour we can in e.g. Cartesian coordinates naturally define functions \f$ f(x,y,z)\f$ and evaluate this function on product space coordinates and have **\f$ x \f$ as the fastest varying coordinate in memory**.
 *
 * @tparam BinarySubroutine Functor with signature: <tt> void ( value_type_g, value_type_y&) </tt> i.e. it reads the first (and second) and writes into the second argument
 * @tparam Functor signature: <tt> value_type_g operator()( value_type_x0, value_type_x1, ...) </tt>
 * @attention Both \c BinarySubroutine and \c Functor must be callable on the device in use. In particular, with CUDA they must be functor tpyes (@b not functions) and their signatures must contain the \__device__ specifier. (s.a. \ref DG_DEVICE)
 * @param y contains result (size of y must match the product of sizes of \f$ x_i\f$)
 * @param f The subroutine, for example \c dg::equals or \c dg::plus_equals, see @ref binary_operators for a collection of predefined functors to use here
 * @param g The functor to evaluate, see @ref functions and @ref variadic_evaluates for a collection of predefined functors to use here
 * @param x0 first input
 * @param xs more input
 * @note all aliases allowed
 * @copydoc hide_naninf
 * @copydoc hide_ContainerType
 *
 * @sa dg::kronecker
 */
template< class ContainerType0, class BinarySubroutine, class Functor, class ContainerType1, class ...ContainerTypes>
inline void kronecker( ContainerType0& y, BinarySubroutine f, Functor g, const ContainerType1& x0, const ContainerTypes& ...xs)
{
    static_assert( ( (dg::is_vector_v<ContainerType0> &&
                      dg::is_vector_v<ContainerType1>) &&
                      ... && dg::is_vector_v<ContainerTypes>),
        "All container types must have a vector data layout (AnyVector)!");
    using vector_type = find_if_t<dg::is_not_scalar, ContainerType1, ContainerType1, ContainerTypes...>;
    using tensor_category  = get_tensor_category<vector_type>;
    static_assert( (
            (dg::is_scalar_or_same_base_category<ContainerType0, tensor_category>::value &&
             dg::is_scalar_or_same_base_category<ContainerType1, tensor_category>::value) &&
            ... && dg::is_scalar_or_same_base_category<ContainerTypes, tensor_category>::value),
        "All container types must be either Scalar or have compatible Vector categories (AnyVector or Same base class)!");
    dg::blas1::detail::doKronecker(tensor_category(), y, f, g, x0, xs...);
}

///@}
}//namespace blas1

/**
 * @brief Generic way to assign the contents of a \c from_ContainerType object to a \c ContainerType object optionally given additional parameters
 *
 * The idea of this function is to convert between types with the same data
 * layout but different execution policies (e.g. from a thrust::host_vector to a thrust::device_vector). If the layout differs, additional parameters can be used
 * to achieve what you want.

 * For example
 * @snippet{trimleft} blas1_t.cpp assign
 *
 * @param from source vector
 * @param to target vector contains a copy of \c from on output (memory is automatically resized if necessary)
 * @param ps additional parameters usable for the transfer operation
 * @note it is possible to assign a \c from_ContainerType to a <tt> std::array<ContainerType, N> </tt>
(all elements are initialized with from_ContainerType) and also a <tt> std::vector<ContainerType></tt> ( the desired size of the \c std::vector must be provided as an additional parameter)
 * @tparam from_ContainerType must have the same data policy derived from \c AnyVectorTag as \c ContainerType (with the exception of \c std::array and \c std::vector) but can have different execution policy
 * @tparam Params in some cases additional parameters that are necessary to assign objects of Type \c ContainerType
 * @copydoc hide_ContainerType
 * @ingroup blas1
 */
template<class from_ContainerType, class ContainerType, class ...Params>
inline void assign( const from_ContainerType& from, ContainerType& to, Params&& ... ps)
{
    dg::detail::doAssign<from_ContainerType, ContainerType, Params...>( from, to, get_tensor_category<from_ContainerType>(), get_tensor_category<ContainerType>(), std::forward<Params>(ps)...);
}

/**
 * @brief Generic way to construct an object of \c ContainerType given a \c from_ContainerType object and optional additional parameters
 *
 * The idea of this function is to convert between types with the same data
 * layout but different execution policies (e.g. from a thrust::host_vector to a thrust::device_vector)
 * If the layout differs, additional parameters can be used to achieve what you want.
 *
 * For example
 * @snippet{trimleft} blas1_t.cpp construct
 * @param from source vector
 * @param ps additional parameters necessary to construct a \c ContainerType object
 * @return \c from converted to the new format (memory is allocated accordingly)
 * @note it is possible to construct a <tt> std::array<ContainerType, N> </tt>
(all elements are initialized with from_ContainerType) and also a <tt> std::vector<ContainerType></tt> ( the desired size of the \c std::vector must be provided as an additional parameter) given a \c from_ContainerType
 * @tparam from_ContainerType must have the same data policy derived from \c AnyVectorTag as \c ContainerType (with the exception of \c std::array and \c std::vector) but can have different execution policy
 * @tparam Params in some cases additional parameters that are necessary to construct objects of Type \c ContainerType
 * @copydoc hide_ContainerType
 * @ingroup blas1
 */
template<class ContainerType, class from_ContainerType, class ...Params>
inline ContainerType construct( const from_ContainerType& from, Params&& ... ps)
{
    return dg::detail::doConstruct<ContainerType, from_ContainerType, Params...>( from, get_tensor_category<ContainerType>(), get_tensor_category<from_ContainerType>(), std::forward<Params>(ps)...);
}

/**
 * @brief \f$ y_I = f(x_{0i_0}, x_{1i_1}, ...) \f$ Memory allocating version of \c dg::blas1::kronecker
 *
 * In a shared memory space with serial execution this function is implemented roughly as in the following pseudo-code
 * @code{.cpp}
 * // assume x0, xs are host vectors
 * size = product( x0.size(), xs.size(), ...)
 * using value_type = decltype( f(x0[0], xs[0]...));
 * thrust::host_vector<value_type> y(size);
 * dg::blas1::kronecker( y, dg::equals(), f, x0, xs...);
 * @endcode
 * @note The return type is inferred from the \c execution_policy and the \c
 * tensor_category of the input vectors.  It is unspecified. It is such that
 * the resulting vector is exactly compatible in a call to
 * <tt> dg::kronecker( result, dg::equals(), f, x0, xs...); </tt>
 *
 * The MPI distributed version of this function is implemented as
 * @code{.cpp}
 * MPI_Comm comm_kron = dg::mpi_cart_kron( x0.communicator(), xs.communicator()...);
 * return MPI_Vector{dg::kronecker( f, x0.data(), xs.data()...), comm_kron}; // a dg::MPI_Vector
 * @endcode
 * @attention In particular this means that in MPI all the communicators in the
 * input argument vectors need to be Cartesian communicators that were created
 * from a common Cartesian root communicator and both root and all sub
 * communicators need to be registered in the dg library through calls to \c
 * dg::register_mpi_cart_sub or \c dg::mpi_cart_sub. Further, the order of
 * input-communicators must match the dimensions in the common root
 * communicator (see \c dg::mpi_cart_kron) i.e. currently **in MPI it is not
 * possible to transpose with this function**
 *
 * The rationale for this behaviour is that:
 *
 * -# the MPI standard has no easy way of finding a common ancestor to
 * Cartesin sub communicators
 * -# the MPI standard has no easy way of re-joining previously split
 * Cartesian communicators
 * -# we want to avoid creating a new communicator every time this
 * function is called.
 * .
 *
 * For example
 * @snippet{trimleft} blas1_t.cpp dg kronecker
 * @tparam Functor signature: <tt> value_type_g operator()( value_type_x0, value_type_x1, ...) </tt>
 * @attention \c Functor must be callable on the device in use. In particular,
 * with CUDA it must be a functor tpye (@b not a function) and its signature
 * must contain the \__device__ specifier. (s.a. \ref DG_DEVICE)
 * @param f The functor to evaluate, see @ref functions and @ref
 * variadic_evaluates for a collection of predefined functors to use here
 * @param x0 first input
 * @param xs more input
 * @return newly allocated result (size of container matches the product of sizes of \f$ x_i\f$)
 *
 * @note all aliases allowed
 * @copydoc hide_naninf
 * @copydoc hide_ContainerType
 *
 * @sa dg::blas1::kronecker dg::mpi_cart_kron
 * @ingroup blas1
 */
template<class ContainerType, class Functor, class ...ContainerTypes>
auto kronecker( Functor f, const ContainerType& x0, const ContainerTypes& ... xs)
{
    using tensor_category  = get_tensor_category<ContainerType>;
    return dg::detail::doKronecker( tensor_category(), f, x0, xs...);
}


} //namespace dg

