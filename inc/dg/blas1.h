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

/*! @brief \f$ x^T y\f$ Binary reproducible Euclidean dot product between two vectors
 *
 * This routine computes \f[ x^T y = \sum_{i=0}^{N-1} x_i y_i \f]
 * @copydoc hide_iterations
 *
For example
@code{.cpp}
dg::DVec two( 100,2), three(100,3);
double result = dg::blas1::dot( two, three); // result = 600 (100*(2*3))
@endcode
 * @attention if one of the input vectors contains \c Inf or \c NaN or the
 * product of the input numbers reaches \c Inf or \c Nan then the behaviour
 * is undefined and the function may throw. See @ref dg::ISNFINITE and @ref
 * dg::ISNSANE in that case
 * @note Our implementation guarantees **binary reproducible** results.
 * The sum is computed with **infinite precision** and the result is rounded
 * to the nearest double precision number.
 * This is possible with the help of an adapted version of the \c dg::exblas library and
* works for single and double precision.

 * @param x Left Container
 * @param y Right Container may alias x
 * @return Scalar product as defined above
 * @note This routine is always executed synchronously due to the
        implicit memcpy of the result. With mpi the result is broadcasted to all processes.
 * @copydoc hide_ContainerType
 */
template< class ContainerType1, class ContainerType2>
inline get_value_type<ContainerType1> dot( const ContainerType1& x, const ContainerType2& y)
{
    std::vector<int64_t> acc = dg::blas1::detail::doDot_superacc( x,y);
    return exblas::cpu::Round(acc.data());
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

For example
@code{.cpp}
//Check if a vector contains Inf or NaN
thrust::device_vector<double> x( 100);
bool hasnan = false;
hasnan = dg::blas1::reduce( x, false, thrust::logical_or<bool>(),
    dg::ISNFINITE<double>());
std::cout << "x contains Inf or NaN "<<std::boolalpha<<hasnan<<"\n";
@endcode
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
    if( std::is_same<ContainerTypeIn, ContainerTypeOut>::value && &source==(const ContainerTypeIn*)&target)
        return;
    dg::blas1::subroutine( dg::equals(), source, target);
}

/*! @brief \f$ x = \alpha x\f$
 *
 * This routine computes \f[ \alpha x_i \f]
 * @copydoc hide_iterations

@code{.cpp}
dg::DVec two( 100,2);
dg::blas1::scal( two,  0.5 )); // result[i] = 1.
@endcode
 * @param alpha Scalar
 * @param x (read/write) x
 * @copydoc hide_naninf
 * @copydoc hide_ContainerType
 */
template< class ContainerType>
inline void scal( ContainerType& x, get_value_type<ContainerType> alpha)
{
    if( alpha == get_value_type<ContainerType>(1))
        return;
    dg::blas1::subroutine( dg::Scal<get_value_type<ContainerType>>(alpha), x );
}

/*! @brief \f$ x = x + \alpha \f$
 *
 * This routine computes \f[ x_i + \alpha \f]
 * @copydoc hide_iterations

@code{.cpp}
dg::DVec two( 100,2);
dg::blas1::plus( two,  3. )); // two[i] = 5.
@endcode
 * @param alpha Scalar
 * @param x (read/write) x
 * @copydoc hide_naninf
 * @copydoc hide_ContainerType
 */
template< class ContainerType>
inline void plus( ContainerType& x, get_value_type<ContainerType> alpha)
{
    if( alpha == get_value_type<ContainerType>(0))
        return;
    dg::blas1::subroutine( dg::Plus<get_value_type<ContainerType>>(alpha), x );
}

/*! @brief \f$ y = \alpha x + \beta y\f$
 *
 * This routine computes \f[ y_i =  \alpha x_i + \beta y_i \f]
 * @copydoc hide_iterations

@code{.cpp}
dg::DVec two( 100,2), three(100,3);
dg::blas1::axpby( 2, two, 3., three); // three[i] = 13 (2*2+3*3)
@endcode
 * @param alpha Scalar
 * @param x ContainerType x may alias y
 * @param beta Scalar
 * @param y (read/write) ContainerType y contains solution on output
 * @copydoc hide_naninf
 * @copydoc hide_ContainerType
 */
template< class ContainerType, class ContainerType1>
inline void axpby( get_value_type<ContainerType> alpha, const ContainerType1& x, get_value_type<ContainerType> beta, ContainerType& y)
{
    using value_type = get_value_type<ContainerType>;
    if( alpha == value_type(0) ) {
        scal( y, beta);
        return;
    }
    if( std::is_same<ContainerType, ContainerType1>::value && &x==(const ContainerType1*)&y){
        dg::blas1::scal( y, (alpha+beta));
        return;
    }
    dg::blas1::subroutine( dg::Axpby<get_value_type<ContainerType>>(alpha, beta),  x, y);
}

/*! @brief \f$ z = \alpha x + \beta y + \gamma z\f$
 *
 * This routine computes \f[ z_i =  \alpha x_i + \beta y_i + \gamma z_i \f]
 * @copydoc hide_iterations

@code{.cpp}
dg::DVec two(100,2), five(100,5), result(100, 12);
dg::blas1::axpbypgz( 2.5, two, 2., five, -3.,result);
// result[i] = -21 (2.5*2+2*5-3*12)
@endcode
 * @param alpha Scalar
 * @param x ContainerType x may alias result
 * @param beta Scalar
 * @param y ContainerType y may alias result
 * @param gamma Scalar
 * @param z (read/write) ContainerType contains solution on output
 * @copydoc hide_naninf
 * @copydoc hide_ContainerType
 */
template< class ContainerType, class ContainerType1, class ContainerType2>
inline void axpbypgz( get_value_type<ContainerType> alpha, const ContainerType1& x, get_value_type<ContainerType> beta, const ContainerType2& y, get_value_type<ContainerType> gamma, ContainerType& z)
{
    using value_type = get_value_type<ContainerType>;
    if( alpha == value_type(0) )
    {
        axpby( beta, y, gamma, z);
        return;
    }
    else if( beta == value_type(0) )
    {
        axpby( alpha, x, gamma, z);
        return;
    }
    if( std::is_same<ContainerType1, ContainerType2>::value && &x==(const ContainerType1*)&y){
        dg::blas1::axpby( alpha+beta, x, gamma, z);
        return;
    }
    else if( std::is_same<ContainerType1, ContainerType>::value && &x==(const ContainerType1*)&z){
        dg::blas1::axpby( beta, y, alpha+gamma, z);
        return;
    }
    else if( std::is_same<ContainerType2, ContainerType>::value && &y==(const ContainerType2*)&z){
        dg::blas1::axpby( alpha, x, beta+gamma, z);
        return;
    }
    dg::blas1::subroutine( dg::Axpbypgz<get_value_type<ContainerType>>(alpha, beta, gamma),  x, y, z);
}

/*! @brief \f$ z = \alpha x + \beta y\f$
 *
 * This routine computes \f[ z_i =  \alpha x_i + \beta y_i \f]
 * @copydoc hide_iterations

@code{.cpp}
dg::DVec two( 100,2), three(100,3), result(100);
dg::blas1::axpby( 2, two, 3., three, result); // result[i] = 13 (2*2+3*3)
@endcode
 * @param alpha Scalar
 * @param x ContainerType x may alias z
 * @param beta Scalar
 * @param y ContainerType y may alias z
 * @param z (write-only) ContainerType z contains solution on output
 * @copydoc hide_naninf
 * @copydoc hide_ContainerType
 */
template< class ContainerType, class ContainerType1, class ContainerType2>
inline void axpby( get_value_type<ContainerType> alpha, const ContainerType1& x, get_value_type<ContainerType> beta, const ContainerType2& y, ContainerType& z)
{
    dg::blas1::evaluate( z , dg::equals(), dg::PairSum(), alpha, x, beta, y);
}

/**
 * @brief \f$ y = \alpha x_1 x_2 + \beta y\f$
 *
 * Multiplies two vectors element by element: \f[ y_i = \alpha x_{1i}x_{2i} + \beta y_i\f]
 * @copydoc hide_iterations

@code{.cpp}
dg::DVec two( 100,2), three( 100,3), result(100,6);
dg::blas1::pointwiseDot(2., two,  three, -4., result );
// result[i] = -12. (2*2*3-4*6)
@endcode
 * @param alpha scalar
 * @param x1 ContainerType x1
 * @param x2 ContainerType x2 may alias x1
 * @param beta scalar
 * @param y (read/write)  ContainerType y contains result on output ( may alias x1 or x2)
 * @copydoc hide_naninf
 * @copydoc hide_ContainerType
 */
template< class ContainerType, class ContainerType1, class ContainerType2>
inline void pointwiseDot( get_value_type<ContainerType> alpha, const ContainerType1& x1, const ContainerType2& x2, get_value_type<ContainerType> beta, ContainerType& y)
{
    if( alpha == get_value_type<ContainerType>(0) ) {
        dg::blas1::scal(y, beta);
        return;
    }
    //not sure this is necessary performance-wise, subroutine does allow aliases
    if( std::is_same<ContainerType, ContainerType1>::value && &x1==(const ContainerType1*)&y){
        dg::blas1::subroutine( dg::AxyPby<get_value_type<ContainerType>>(alpha,beta), x2, y );

        return;
    }
    if( std::is_same<ContainerType, ContainerType2>::value && &x2==(const ContainerType2*)&y){
        dg::blas1::subroutine( dg::AxyPby<get_value_type<ContainerType>>(alpha,beta), x1, y );

        return;
    }
    dg::blas1::subroutine( dg::PointwiseDot<get_value_type<ContainerType>>(alpha,beta), x1, x2, y );
}

/*! @brief \f$ y = x_1 x_2 \f$
*
* Multiplies two vectors element by element: \f[ y_i = x_{1i}x_{2i}\f]
* @copydoc hide_iterations

@code{.cpp}
dg::DVec two( 100,2), three( 100,3), result(100);
dg::blas1::pointwiseDot( two,  three, result ); // result[i] = 6.
@endcode
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

@code{.cpp}
dg::DVec two( 100,2), three( 100,3), four(100,4), result(100,6);
dg::blas1::pointwiseDot(2., two,  three, four, -4., result );
// result[i] = 24. (2*2*3*4-4*6)
@endcode
* @param alpha scalar
* @param x1 ContainerType x1
* @param x2 ContainerType x2 may alias x1
* @param x3 ContainerType x3 may alias x1 and/or x2
* @param beta scalar
* @param y  (read/write) ContainerType y contains result on output ( may alias x1,x2 or x3)
* @copydoc hide_naninf
* @copydoc hide_ContainerType
*/
template< class ContainerType, class ContainerType1, class ContainerType2, class ContainerType3>
inline void pointwiseDot( get_value_type<ContainerType> alpha, const ContainerType1& x1, const ContainerType2& x2, const ContainerType3& x3, get_value_type<ContainerType> beta, ContainerType& y)
{
    if( alpha == get_value_type<ContainerType>(0) ) {
        dg::blas1::scal(y, beta);
        return;
    }
    dg::blas1::subroutine( dg::PointwiseDot<get_value_type<ContainerType>>(alpha,beta), x1, x2, x3, y );
}

/**
* @brief \f$ y = \alpha x_1/ x_2 + \beta y \f$
*
* Divides two vectors element by element: \f[ y_i = \alpha x_{1i}/x_{2i} + \beta y_i \f]

* @copydoc hide_iterations

@code{.cpp}
dg::DVec two( 100,2), three( 100,3), result(100,1);
dg::blas1::pointwiseDivide( 3, two,  three, 5, result );
// result[i] = 7 (3*2/3+5*1)
@endcode
* @param alpha scalar
* @param x1 ContainerType x1
* @param x2 ContainerType x2 may alias x1
* @param beta scalar
* @param y  (read/write) ContainerType y contains result on output ( may alias x1 and/or x2)
* @copydoc hide_naninf
* @copydoc hide_ContainerType
*/
template< class ContainerType, class ContainerType1, class ContainerType2>
inline void pointwiseDivide( get_value_type<ContainerType> alpha, const ContainerType1& x1, const ContainerType2& x2, get_value_type<ContainerType> beta, ContainerType& y)
{
    if( alpha == get_value_type<ContainerType>(0) ) {
        dg::blas1::scal(y, beta);
        return;
    }
    if( std::is_same<ContainerType, ContainerType1>::value && &x1==(const ContainerType1*)&y){
        dg::blas1::subroutine( dg::PointwiseDivide<get_value_type<ContainerType>>(alpha,beta), x2, y );

        return;
    }
    dg::blas1::subroutine( dg::PointwiseDivide<get_value_type<ContainerType>>(alpha, beta), x1, x2, y );
}

/**
* @brief \f$ y = x_1/ x_2\f$
*
* Divides two vectors element by element: \f[ y_i = x_{1i}/x_{2i}\f]
* @copydoc hide_iterations

@code{.cpp}
dg::DVec two( 100,2), three( 100,3), result(100);
dg::blas1::pointwiseDivide( two,  three, result );
// result[i] = -0.666... (2/3)
@endcode
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

@code{.cpp}
dg::DVec two(100,2), three(100,3), four(100,5), five(100,5), result(100,6);
dg::blas1::pointwiseDot(2., two,  three, -4., four, five, 2., result );
// result[i] = -56.
@endcode
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
*/
template<class ContainerType, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerType4>
void pointwiseDot(  get_value_type<ContainerType> alpha, const ContainerType1& x1, const ContainerType2& y1,
                    get_value_type<ContainerType> beta,  const ContainerType3& x2, const ContainerType4& y2,
                    get_value_type<ContainerType> gamma, ContainerType & z)
{
    using value_type = get_value_type<ContainerType>;
    if( alpha==value_type(0)){
        pointwiseDot( beta, x2,y2, gamma, z);
        return;
    }
    else if( beta==value_type(0)){
        pointwiseDot( alpha, x1,y1, gamma, z);
        return;
    }
    dg::blas1::subroutine( dg::PointwiseDot<get_value_type<ContainerType>>(alpha, beta, gamma), x1, y1, x2, y2, z );
}

/*! @brief \f$ y = op(x)\f$
 *
 * This routine computes \f[ y_i = op(x_i) \f]
 * @copydoc hide_iterations

@code{.cpp}
dg::DVec two( 100,2), result(100);
dg::blas1::transform( two, result, dg::EXP<double>());
// result[i] = 7.389056... (e^2)
@endcode
 * @param x ContainerType x may alias y
 * @param y (write-only) ContainerType y contains result, may alias x
 * @param op unary %Operator to use on every element
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
@code{.cpp}
double function( double x, double y) {
    return sin(x)*sin(y);
}
dg::HVec pi2(20, M_PI/2.), pi3( 20, 3*M_PI/2.), result(20, 0);
dg::blas1::evaluate( result, dg::equals(), function, pi2, pi3);
// result[i] = sin(M_PI/2.)*sin(3*M_PI/2.) = -1
@endcode
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

template< class ContainerType1, class ContainerType2>
inline std::vector<int64_t> doDot_superacc( const ContainerType1& x, const ContainerType2& y)
{
    static_assert( all_true<
            dg::is_vector<ContainerType1>::value,
            dg::is_vector<ContainerType2>::value>::value,
        "All container types must have a vector data layout (AnyVector)!");
    using vector_type = find_if_t<dg::is_not_scalar, ContainerType1, ContainerType1, ContainerType2>;
    using tensor_category  = get_tensor_category<vector_type>;
    static_assert( all_true<
            dg::is_scalar_or_same_base_category<ContainerType1, tensor_category>::value,
            dg::is_scalar_or_same_base_category<ContainerType2, tensor_category>::value
            >::value,
        "All container types must be either Scalar or have compatible Vector categories (AnyVector or Same base class)!");
    return doDot_superacc( x, y, tensor_category());
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
@code{.cpp}
struct Routine{
DG_DEVICE
void operator()( double x, double y, double& z){
   z = 7*x+y + z ;
}
};
dg::DVec two( 100,2), four(100,4);
dg::blas1::subroutine( Routine(), two, 3., four);
// four[i] now has the value 21 (7*2+3+4)
@endcode

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
    static_assert( all_true<
            dg::is_vector<ContainerType>::value,
            dg::is_vector<ContainerTypes>::value...>::value,
        "All container types must have a vector data layout (AnyVector)!");
    using vector_type = find_if_t<dg::is_not_scalar, ContainerType, ContainerType, ContainerTypes...>;
    using tensor_category  = get_tensor_category<vector_type>;
    static_assert( all_true<
            dg::is_scalar_or_same_base_category<ContainerType, tensor_category>::value,
            dg::is_scalar_or_same_base_category<ContainerTypes, tensor_category>::value...
            >::value,
        "All container types must be either Scalar or have compatible Vector categories (AnyVector or Same base class)!");
    //using basic_tag_type  = std::conditional_t< all_true< is_scalar<ContainerType>::value, is_scalar<ContainerTypes>::value... >::value, AnyScalarTag , AnyVectorTag >;
    dg::blas1::detail::doSubroutine(tensor_category(), f, std::forward<ContainerType>(x), std::forward<ContainerTypes>(xs)...);
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
 * @code{.cpp}
dg::HVec host( 100, 1.);
dg::DVec device(100);
dg::assign( host, device );
//let us construct a std::vector of 3 dg::DVec from a host vector
std::vector<dg::DVec> device_vec(3);
dg::assign( host, device_vec, 3);
 * @endcode
 * @param from source vector
 * @param to target vector contains a copy of \c from on output (memory is automatically resized if necessary)
 * @param ps additional parameters usable for the transfer operation
 * @note it is possible to assign a \c from_ContainerType to a <tt> std::array<ContainerType, N> </tt>
(all elements are initialized with from_ContainerType) and also a <tt> std::vector<ContainerType></tt> ( the desired size of the \c std::vector must be provided as an additional parameter)
 * @tparam from_ContainerType must have the same data policy derived from \c AnyVectorTag as \c ContainerType (with the exception of \c std::array and \c std::vector) but can have different execution policy
 * @tparam Params in some cases additional parameters that are necessary to assign objects of Type \c ContainerType
 * @copydoc hide_ContainerType
 * @ingroup backend
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

 * For example
 * @code{.cpp}
dg::HVec host( 100, 1.);
dg::DVec device = dg::construct<dg::DVec>( host );
std::array<dg::DVec, 3> device_arr = dg::construct<std::array<dg::DVec, 3>>( host );
//let us construct a std::vector of 3 dg::DVec from a host vector
std::vector<dg::DVec> device_vec = dg::construct<std::vector<dg::DVec>>( host, 3);
 * @endcode
 * @param from source vector
 * @param ps additional parameters necessary to construct a \c ContainerType object
 * @return \c from converted to the new format (memory is allocated accordingly)
 * @note it is possible to construct a <tt> std::array<ContainerType, N> </tt>
(all elements are initialized with from_ContainerType) and also a <tt> std::vector<ContainerType></tt> ( the desired size of the \c std::vector must be provided as an additional parameter) given a \c from_ContainerType
 * @tparam from_ContainerType must have the same data policy derived from \c AnyVectorTag as \c ContainerType (with the exception of \c std::array and \c std::vector) but can have different execution policy
 * @tparam Params in some cases additional parameters that are necessary to construct objects of Type \c ContainerType
 * @copydoc hide_ContainerType
 * @ingroup backend
 */
template<class ContainerType, class from_ContainerType, class ...Params>
inline ContainerType construct( const from_ContainerType& from, Params&& ... ps)
{
    return dg::detail::doConstruct<ContainerType, from_ContainerType, Params...>( from, get_tensor_category<ContainerType>(), get_tensor_category<from_ContainerType>(), std::forward<Params>(ps)...);
}


} //namespace dg

