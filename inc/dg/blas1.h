#pragma once

#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "backend/vector_traits.h"
#include "backend/blas1_dispatch_shared.h"
#include "backend/blas1_array.h"
#include "backend/vector_traits_cusp.h"
#ifdef MPI_VERSION
#include "backend/mpi_vector.h"
#include "backend/blas1_dispatch_mpi.h"
#endif
#include "backend/blas1_dispatch_vector.h"
#include "subroutines.h"

/*!@file
 *
 * Basic linear algebra level 1 functions (functions that only involve vectors and not matrices)
 */

namespace dg{

/*! @brief BLAS Level 1 routines
 *
 * @ingroup blas1
 * Only those routines that are actually called need to be implemented.
 * @note successive calls to blas routines are executed sequentially
 * @note A manual synchronization of threads or devices is never needed in an application
 * using these functions. All functions returning a value block until the value is ready.
 */
namespace blas1
{

///@addtogroup blas1
///@{

/**
 * @class hide_iterations
 * where \c i iterates over @b all elements inside the ContainerType. If \c ContainerType has the \c VectorVectorTag, \c i recursively loops over all entries.
 * If the ContainerType sizes do not match, the result is undefined.
 * The compiler chooses the implementation of this function based on the execution policy tag of the ContainerTypes.
 */

/**
 * @brief y=x; Generic way to copy-construct/assign-to an object of \c to_ContainerType type from a different \c from_ContainerType type
 *
 * The idea of this function is to convert between objects with the same data
 * layout but different execution policies (e.g. from CPU to GPU, or sequential
 * to parallel execution)
 * @copydoc hide_ContainerType
 * @tparam to_ContainerType another ContainerType type, must have the same data policy derived from \c AnyVectorTag as \c ContainerType (with the exception of \c std::array) but can have different exectuion policy
 * @param src source
 * @return x converted to the new format
 * @note since this function is quite often used there is a higher level alias \c dg::transfer(From&&)
 * @note it is possible to transfer a ContainerType to a <tt> std::array<ContainerType, N> </tt>(all elements are initialized to ContainerType) but not a <tt> std::vector<ContainerType></tt> (since the desired size of the \c std::vector cannot be known)

 * For example
@code
dg::DVec device = dg::tansfer<dg::DVec>( dg::evaluate(dg::one, grid));
std::array<dg::DVec, 3> device_arr = dg::transfer<std::array<dg::DVec, 3>>( dg::evaluate( dg::one, grid));
@endcode
 */
template<class to_ContainerType, class from_ContainerType>
inline to_ContainerType transfer( const from_ContainerType& src)
{
    return dg::blas1::detail::doTransfer<to_ContainerType, from_ContainerType>( src, get_vector_category<to_ContainerType>(), get_vector_category<from_ContainerType>());
}

/**
 * @brief y=x; Generic way to copy an object of \c from_ContainerType type to an object of different \c to_ContainerType type
 *
 * The idea of this function is to convert between objects with the same data
 * layout but different execution policies (e.g. from CPU to GPU, or sequential
 * to parallel execution)
 * @copydoc hide_ContainerType
 * @tparam to_ContainerType another ContainerType type, must have the same data policy derived from \c AnyVectorTag as \c ContainerType (with the exception of \c std::array) but can have different exectuion policy
 * @param source source
 * @param target sink
 * @note y gets resized properly
 * @note since this function is quite often used there is a higher level alias \c dg::transfer(From&&,To&&)
 * @note it is possible to transfer a ContainerType to a <tt> std::array<ContainerType, N> </tt>(all elements are initialized to ContainerType) but not a <tt> std::vector<ContainerType></tt> (since the desired size of the \c std::vector cannot be known)
 *
 * For example
@code
dg::HVec host = dg::evaluate( dg::one, grid);
dg::DVec device;
dg::transfer( host, device); //device now equals host
std::array<dg::DVec, 3> device_arr;
dg::transfer( host, device_arr); //every element of device_arr now equals host
@endcode

 */
template<class from_ContainerType, class to_ContainerType>
inline void transfer( const from_ContainerType& source, to_ContainerType& target)
{
    target = dg::blas1::transfer<to_ContainerType, from_ContainerType>( source);
}

/*! @brief \f$ x^T y\f$ Binary reproducible Euclidean dot product between two vectors
 *
 * This routine computes \f[ x^T y = \sum_{i=0}^{N-1} x_i y_i \f]
 * @copydoc hide_iterations
 * Our implementation guarantees binary reproducible results.
 * The sum is computed with infinite precision and the result is rounded
 * to the nearest double precision number.
 * This is possible with the help of an adapted version of the \c ::exblas library.
 * @copydoc hide_ContainerType
 * @param x Left ContainerType
 * @param y Right ContainerType may alias x
 * @return Scalar product as defined above
 * @note This routine is always executed synchronously due to the
        implicit memcpy of the result. With mpi the result is broadcasted to all processes

For example
@code
dg::DVec two( 100,2), three(100,3);
double temp = dg::blas1::dot( two, three); //temp = 30 (5*(2*3))
@endcode
 */
template< class ContainerType1, class ContainerType2>
inline get_value_type<ContainerType1> dot( const ContainerType1& x, const ContainerType2& y)
{
    return dg::blas1::detail::doDot( x, y, get_vector_category<ContainerType1>() );
}

/**
 * @brief \f$ f(x_0, x_1, ...)\f$
 *
 * This routine evaluates an arbitrary user-defined subroutine \c f with an arbitrary number of arguments \f$ x_s\f$ elementwise
 * \f[ f(x_{0i}, x_{1i}, ...)  \f]
 * @copydoc hide_iterations
 * @tparam Subroutine
 * @tparam ContainerType
 * @tparam ...ContainerTypes
 * @param f the subroutine
 * @param x the first argument
 * @param xs other arguments
 *
@code
void routine( double x, double y, double& z){
   z = 7*x+y + z ;
}
dg::DVec two( 100,2), three(100,3), four(100,4);
dg::blas1::subroutine( routine, two, three, four);
//four[i] now has the value 21 (7*2+3+4)
@endcode
@note if you do not think that this function is plain magic you haven't looked at it
long enough. This function can compute @b any trivial parallel expression for @b any
number of inputs and outputs. In this sense it replaces all other \c blas1 functions
except the scalar product, which is not trivial parallel.
 */
template< class Subroutine, class ContainerType, class ...ContainerTypes>
inline void subroutine( Subroutine f, ContainerType&& x, ContainerTypes&&... xs)
{
    dg::blas1::detail::doSubroutine( get_vector_category<ContainerType>(), f, std::forward<ContainerType>(x), std::forward<ContainerTypes>(xs)...);
    return;
}
/**
 * @brief \f$ y=x \f$
 *
 * explicit pointwise assignment \f$ y_i = x_i\f$
 * @copydoc hide_iterations
 * @copydoc hide_ContainerType
 * @param x in
 * @param y out
 * @note in contrast to the \c blas1::transfer functions the copy function is
 * explicitly parallel and thus works only on types with same execution
 * policy
 */
template<class ContainerType_in, class ContainerType_out>
inline void copy( const ContainerType_in& x, ContainerType_out& y){
    dg::blas1::subroutine( dg::equals(), y, x );
    return;
}

/*! @brief \f$ x = \alpha x\f$
 *
 * This routine computes \f[ \alpha x_i \f]
 * @copydoc hide_iterations
 * @copydoc hide_ContainerType
 * @param alpha Scalar
 * @param x ContainerType x

@code
dg::DVec two( 100,2);
dg::blas1::scal( two,  0.5 )); //result[i] = 1.
@endcode
 */
template< class ContainerType>
inline void scal( ContainerType& x, get_value_type<ContainerType> alpha)
{
    if( alpha == get_value_type<ContainerType>(1))
        return;
    dg::blas1::subroutine( dg::Scal<get_value_type<ContainerType>>(alpha), x );
    return;
}

/*! @brief \f$ x = x + \alpha \f$
 *
 * This routine computes \f[ x_i + \alpha \f]
 * @copydoc hide_iterations
 * @copydoc hide_ContainerType
 * @param alpha Scalar
 * @param x ContainerType x

@code
dg::DVec two( 100,2);
dg::blas1::plus( two,  2. )); //result[i] = 4.
@endcode
 */
template< class ContainerType>
inline void plus( ContainerType& x, get_value_type<ContainerType> alpha)
{
    if( alpha == get_value_type<ContainerType>(0))
        return;
    dg::blas1::subroutine( dg::Plus<get_value_type<ContainerType>>(alpha), x );
    return;
}

/*! @brief \f$ y = \alpha x + \beta y\f$
 *
 * This routine computes \f[ y_i =  \alpha x_i + \beta y_i \f]
 * @copydoc hide_iterations
 * @copydoc hide_ContainerType
 * @param alpha Scalar
 * @param x ContainerType x may alias y
 * @param beta Scalar
 * @param y ContainerType y contains solution on output

@code
dg::DVec two( 100,2), three(100,3);
dg::blas1::axpby( 2, two, 3., three); //three[i] = 13 (2*2+3*3)
@endcode
 */
template< class ContainerType, class ContainerType1>
inline void axpby( get_value_type<ContainerType> alpha, const ContainerType1& x, get_value_type<ContainerType> beta, ContainerType& y)
{
    using value_type = get_value_type<ContainerType>;
    if( alpha == value_type(0) ) {
        scal( y, beta);
        return;
    }
    dg::blas1::subroutine( dg::Axpby<get_value_type<ContainerType>>(alpha, beta),  x, y);
    return;
}

/*! @brief \f$ z = \alpha x + \beta y + \gamma z\f$
 *
 * This routine computes \f[ z_i =  \alpha x_i + \beta y_i + \gamma z_i \f]
 * @copydoc hide_iterations
 * @copydoc hide_ContainerType
 * @param alpha Scalar
 * @param x ContainerType x may alias result
 * @param beta Scalar
 * @param y ContainerType y may alias result
 * @param gamma Scalar
 * @param z ContainerType contains solution on output

@code
dg::DVec two(100,2), five(100,5), result(100, 12);
dg::blas1::axpbypgz( 2.5, two, 2., five, -3.,result);
//result[i] = -21 (2.5*2+2*5-3*12)
@endcode
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
    dg::blas1::subroutine( dg::Axpbypgz<get_value_type<ContainerType>>(alpha, beta, gamma),  x, y, z);
    return;
}

/*! @brief \f$ z = \alpha x + \beta y\f$
 *
 * This routine computes \f[ z_i =  \alpha x_i + \beta y_i \f]
 * @copydoc hide_iterations
 * @copydoc hide_ContainerType
 * @param alpha Scalar
 * @param x ContainerType x may alias z
 * @param beta Scalar
 * @param y ContainerType y may alias z
 * @param z ContainerType z contains solution on output

@code
dg::DVec two( 100,2), three(100,3), result(100);
dg::blas1::axpby( 2, two, 3., three, result); //result[i] = 13 (2*2+3*3)
@endcode
 */
template< class ContainerType, class ContainerType1, class ContainerType2>
inline void axpby( get_value_type<ContainerType> alpha, const ContainerType1& x, get_value_type<ContainerType> beta, const ContainerType2& y, ContainerType& z)
{
    dg::blas1::axpbypgz( alpha, x,  beta, y, 0., z);
    return;
}

/*! @brief \f$ y = f(y, g(x_0,x_1,...)\f$
 *
 * This routine elementwise evaluates \f[ f(y_i , g(x_{0i}, x_{1i}, ...)) \f]
 * @copydoc hide_iterations
 * @copydoc hide_ContainerType
 * @tparam BinarySubroutine Functor with signature: \c void \c operator()( value_type_y&, value_type_g) i.e. it writes into the first and reads from its second argument
 * @tparam Functor signature: \c value_type_g \c operator()( value_type_x0, value_type_x1, ...)
 * @param y contains result
 * @param f The subroutine
 * @param g The functor to evaluate
 * @param x0 first input
 * @param xs more input
 * @note the Functor must be callable on the device in use. In particular, with CUDA its signature must contain the \__device__ specifier. (s.a. \ref DG_DEVICE)
 * @note all aliases allowed
 *
@code
double function( double x, double y) {return sin(x)*sin(y);}
dg::HVec pi2(20, M_PI/2.), pi3( 20, 3*M_PI/2.), result(20, 0);
dg::blas1::evaluate( result, dg::equals(), function, pi2, pi3);
//result[i] =  -1. (sin(M_PI/2.)*sin(3*M_PI/2.))
@endcode
 */
template< class ContainerType, class BinarySubroutine, class Functor, class ContainerType0, class ...ContainerTypes>
inline void evaluate( ContainerType& y, BinarySubroutine f, Functor g, const ContainerType0& x0, const ContainerTypes& ...xs)
{
    dg::blas1::subroutine( dg::Evaluate<BinarySubroutine, Functor>(f,g), y, x0, xs...);
    return;
}

/*! @brief \f$ y = op(x)\f$
 *
 * This routine computes \f[ y_i = op(x_i) \f]
 * @copydoc hide_iterations
 * @copydoc hide_ContainerType
 * @tparam UnaryOp Functor with signature: \c value_type \c operator()( value_type)
 * @param x ContainerType x may alias y
 * @param y ContainerType y contains result, may alias x
 * @param op unary Operator to use on every element
 * @note the Functor must be callable on the device in use. In particular, with CUDA its signature must contain the \__device__ specifier. (s.a. \ref DG_DEVICE)

@code
dg::DVec two( 100,2), result(100);
dg::blas1::transform( two, result, dg::EXP<double>());
//result[i] = 7.389056... (e^2)
@endcode
 */
template< class ContainerType, class ContainerType1, class UnaryOp>
inline void transform( const ContainerType1& x, ContainerType& y, UnaryOp op )
{
    dg::blas1::evaluate( y, dg::equals(), op, x);
}

/*! @brief \f$ y = x_1 x_2 \f$
*
* Multiplies two vectors element by element: \f[ y_i = x_{1i}x_{2i}\f]
* @copydoc hide_iterations
* @copydoc hide_ContainerType
* @param x1 ContainerType x1
* @param x2 ContainerType x2 may alias x1
* @param y  ContainerType y contains result on output ( may alias x1 or x2)

@code
dg::DVec two( 100,2), three( 100,3), result(100);
dg::blas1::pointwiseDot( two,  three, result ); //result[i] = 6.
@endcode
*/
template< class ContainerType, class ContainerType1, class ContainerType2>
inline void pointwiseDot( const ContainerType1& x1, const ContainerType2& x2, ContainerType& y)
{
    dg::blas1::subroutine( dg::PointwiseDot<get_value_type<ContainerType>>(1,0), x1, x2, y );
    return;
}

/**
* @brief \f$ y = \alpha x_1 x_2 + \beta y\f$
*
* Multiplies two vectors element by element: \f[ y_i = \alpha x_{1i}x_{2i} + \beta y_i\f]
* @copydoc hide_iterations
* @copydoc hide_ContainerType
* @param alpha scalar
* @param x1 ContainerType x1
* @param x2 ContainerType x2 may alias x1
* @param beta scalar
* @param y  ContainerType y contains result on output ( may alias x1 or x2)

@code
dg::DVec two( 100,2), three( 100,3), result(100,6);
dg::blas1::pointwiseDot(2., two,  three, -4., result );
//result[i] = -12. (2*2*3-4*6)
@endcode
*/
template< class ContainerType, class ContainerType1, class ContainerType2>
inline void pointwiseDot( get_value_type<ContainerType> alpha, const ContainerType1& x1, const ContainerType2& x2, get_value_type<ContainerType> beta, ContainerType& y)
{
    if( alpha == get_value_type<ContainerType>(0) ) {
        dg::blas1::scal(y, beta);
        return;
    }
    dg::blas1::subroutine( dg::PointwiseDot<get_value_type<ContainerType>>(alpha,beta), x1, x2, y );
}

/**
* @brief \f$ y = \alpha x_1 x_2 x_3 + \beta y\f$
*
* Multiplies three vectors element by element: \f[ y_i = \alpha x_{1i}x_{2i}x_{3i} + \beta y_i\f]
* @copydoc hide_iterations
* @copydoc hide_ContainerType
* @param alpha scalar
* @param x1 ContainerType x1
* @param x2 ContainerType x2 may alias x1
* @param x3 ContainerType x3 may alias x1 and/or x2
* @param beta scalar
* @param y  ContainerType y contains result on output ( may alias x1,x2 or x3)

@code
dg::DVec two( 100,2), three( 100,3), four(100,4), result(100,6);
dg::blas1::pointwiseDot(2., two,  three, four, -4., result );
//result[i] = 24. (2*2*3*4-4*6)
@endcode
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
* @brief \f$ y = x_1/ x_2\f$
*
* Divides two vectors element by element: \f[ y_i = x_{1i}/x_{2i}\f]
* @copydoc hide_iterations
* @copydoc hide_ContainerType
* @param x1 ContainerType x1
* @param x2 ContainerType x2 may alias x1
* @param y  ContainerType y contains result on output ( may alias x1 and/or x2)

@code
dg::DVec two( 100,2), three( 100,3), result(100);
dg::blas1::pointwiseDivide( two,  three, result );
//result[i] = -0.666... (2/3)
@endcode
*/
template< class ContainerType, class ContainerType1, class ContainerType2>
inline void pointwiseDivide( const ContainerType1& x1, const ContainerType2& x2, ContainerType& y)
{
    dg::blas1::subroutine( dg::PointwiseDivide<get_value_type<ContainerType>>(1,0), x1, x2, y );
    return;
}
/**
* @brief \f$ y = \alpha x_1/ x_2 + \beta y \f$
*
* Divides two vectors element by element: \f[ y_i = \alpha x_{1i}/x_{2i} + \beta y_i \f]

* @copydoc hide_iterations
* @copydoc hide_ContainerType
* @param alpha scalar
* @param x1 ContainerType x1
* @param x2 ContainerType x2 may alias x1
* @param beta scalar
* @param y  ContainerType y contains result on output ( may alias x1 and/or x2)

@code
dg::DVec two( 100,2), three( 100,3), result(100,1);
dg::blas1::pointwiseDivide( 3, two,  three, 5, result );
//result[i] = 7 (3*2/3+5*1)
@endcode
*/
template< class ContainerType, class ContainerType1, class ContainerType2>
inline void pointwiseDivide( get_value_type<ContainerType> alpha, const ContainerType1& x1, const ContainerType2& x2, get_value_type<ContainerType> beta, ContainerType& y)
{
    if( alpha == get_value_type<ContainerType>(0) ) {
        dg::blas1::scal(y, beta);
        return;
    }
    dg::blas1::subroutine( dg::PointwiseDivide<get_value_type<ContainerType>>(alpha, beta), x1, x2, y );
}

/**
* @brief \f$ z = \alpha x_1x_2 + \beta x_2y_2 + \gamma z\f$
*
* Multiplies and adds vectors element by element: \f[ z_i = \alpha x_{1i}y_{1i} + \beta x_{2i}y_{2i} + \gamma z_i \f]
* @copydoc hide_iterations
* @copydoc hide_ContainerType
* @param alpha scalar
* @param x1 ContainerType x1
* @param y1 ContainerType y1
* @param beta scalar
* @param x2 ContainerType x2
* @param y2 ContainerType y2
* @param gamma scalar
* @param z  ContainerType z contains result on output
* @note all aliases are allowed

@code
dg::DVec two(100,2), three(100,3), four(100,5), five(100,5), result(100,6);
dg::blas1::pointwiseDot(2., two,  three, -4., four, five, 2., result );
//result[i] = -56.
@endcode
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
    return;
}
///@}
}//namespace blas1

/**
* @brief High level alias for \c dg::blas1::transfer(const from_ContainerType&,to_ContainerType&)
*
* @ingroup misc
* @tparam From source type
* @tparam To target type
* @param arg1 source object
* @param arg2 target object
*/
template< class From, class To>
void transfer(From&& arg1, To&& arg2){
    blas1::transfer( std::forward<From>(arg1), std::forward<To>(arg2));
}
/**
* @brief High level alias for \c dg::blas1::transfer(const from_ContainerType&)
*
* @ingroup misc
* @tparam To target type
* @tparam From source type
* @param arg source object
* @return target object
*/
template< class To, class From>
To transfer(From&& arg){
    return blas1::transfer<To,From>( std::forward<From>(arg));
}

} //namespace dg

