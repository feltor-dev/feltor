#pragma once

#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "backend/vector_traits.h"
#include "backend/thrust_vector_blas.cuh"
#include "backend/blas1_array.h"
#include "backend/cusp_vector_blas.h"
#ifdef MPI_VERSION
#include "backend/mpi_vector.h"
#include "backend/mpi_vector_blas.h"
#endif
#include "backend/std_vector_blas.cuh"
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
 * @brief y=x; Generic way to copy/construct and/or convert a container type from a different container type (e.g. from CPU to GPU, or double to float, etc.)
 *
 * @copydoc hide_container
 * @tparam other_container another container type, must have the same data policy derived from \c AnyVectorTag as \c container
 * @param x source
 * @return x converted to the new format
 * @note since this function is quite often used there is a higher level alias \c dg::transfer
 * @note it is possible to transfer a container to a std::array<container, N> (all elements are initialized to container) but not a std::vector<container> (since the desired size of the std::vector cannot be known)

 * For example
 * @code
 dg::DVec device = dg::tansfer<dg::DVec>( dg::evaluate(dg::one, grid));
 std::array<dg::DVec, 3> device_arr = dg::transfer<std::array<dg::DVec, 3>>( dg::evaluate( dg::one, grid));
 * @endcode
 */
template<class container, class other_container>
inline container transfer( const other_container& x)
{
    return dg::blas1::detail::doTransfer<container, other_container>( x, get_vector_category<container>(), get_vector_category<other_container>());
}

/**
 * @brief y=x; Generic way to copy/assign and/or convert a container type to a different container type (e.g. from CPU to GPU, or double to float, etc.)
 *
 * @copydoc hide_container
 * @tparam other_container another container type, must have the same data policy derived from \c AnyVectorTag as \c container
 * @param x source
 * @param y sink
 * @note y gets resized properly
 * @note since this function is quite often used there is a higher level alias \c dg::transfer
 * @note it is possible to transfer a container to a std::array<container, N> (all elements are initialized to container) but not a std::vector<container> (since the desired size of the std::vector cannot be known)
 *
 * For example
 * @code
 dg::HVec host = dg::evaluate( dg::one, grid);
 dg::DVec device;
 dg::transfer( host, device); //device now equals host
 std::array<dg::DVec, 3> device_arr;
 dg::transfer( host, device_arr); //every element of device_arr now equals host
 * @endcode

 */
template<class container, class other_container>
inline void transfer( const other_container& x, container& y)
{
    y = dg::blas1::transfer<container, other_container>( x);
}


/**
 * @brief y=x; Invoke assignment operator
 *
 * Same as y=x
 * @tparam Assignable any assignable type
 * @param x in
 * @param y out
 */
template<class Assignable>
inline void copy( const Assignable& x, Assignable& y){y=x;}

/*! @brief \f$ x^T y\f$; Binary reproducible Euclidean dot product between two containers
 *
 * This routine computes \f[ x^T y = \sum_{i=0}^{N-1} x_i y_i \f]  i iterates over @b all elements inside the container. If \c container has the \c VectorVectorTag, i recursively loops over all entries. If the container sizes
 * do not match, the result is undefined.
 * Our implementation guarantees binary reproducible results.
 * The sum is computed with infinite precision and the result is rounded
 * to the nearest double precision number.
 * This is possible with the help of an adapted version of the \c ::exblas library.
 * @copydoc hide_container
 * @param x Left container
 * @param y Right container may alias x
 * @return Scalar product as defined above
 * @note This routine is always executed synchronously due to the
        implicit memcpy of the result. With mpi the result is broadcasted to all processes

For example
@code
    dg::DVec two( 100,2), three(100,3);
    double temp = dg::blas1::dot( two, three); //temp = 30 (5*(2*3))
@endcode
 */
template< class container>
inline get_value_type<container> dot( const container& x, const container& y)
{
    return dg::blas1::detail::doDot( x, y, get_vector_category<container>() );
}

/*! @brief \f$ y = \alpha x + \beta y\f$
 *
 * This routine computes \f[ y_i =  \alpha x_i + \beta y_i \f]  i iterates over @b all elements inside the container. If \c container has the \c VectorVectorTag, i recursively loops over all entries. If the container sizes
 * do not match, the result is undefined.
 * @copydoc hide_container
 * @param alpha Scalar
 * @param x container x may alias y
 * @param beta Scalar
 * @param y container y contains solution on output

@code
    dg::DVec two( 100,2), three(100,3);
    dg::blas1::axpby( 2, two, 3., three); //three[i] = 13 (2*2+3*3)
@endcode
 */
template< class container>
inline void axpby( get_value_type<container> alpha, const container& x, get_value_type<container> beta, container& y)
{
    dg::blas1::detail::doAxpby( alpha, x, beta, y, get_vector_category<container>() );
    return;
}

/*! @brief \f$ z = \alpha x + \beta y\f$
 *
 * This routine computes \f[ z_i =  \alpha x_i + \beta y_i \f]  i iterates over @b all elements inside the container. If \c container has the \c VectorVectorTag, i recursively loops over all entries. If the container sizes
 * do not match, the result is undefined.

 * @copydoc hide_container
 * @param alpha Scalar
 * @param x container x may alias z
 * @param beta Scalar
 * @param y container y may alias z
 * @param z container z contains solution on output

@code
    dg::DVec two( 100,2), three(100,3), result(100);
    dg::blas1::axpby( 2, two, 3., three, result); //result[i] = 13 (2*2+3*3)
@endcode
 */
template< class container>
inline void axpby( get_value_type<container> alpha, const container& x, get_value_type<container> beta, const container& y, container& z)
{
    dg::blas1::detail::doAxpbypgz( alpha, x, beta, y, 0, z, get_vector_category<container>() );
    return;
}

/*! @brief \f$ z = \alpha x + \beta y + \gamma z\f$
 *
 * This routine computes \f[ z_i =  \alpha x_i + \beta y_i + \gamma z_i \f]  i iterates over @b all elements inside the container. If \c container has the \c VectorVectorTag, i recursively loops over all entries. If the container sizes
 * do not match, the result is undefined.

 * @copydoc hide_container
 * @param alpha Scalar
 * @param x container x may alias result
 * @param beta Scalar
 * @param y container y may alias result
 * @param gamma Scalar
 * @param z container contains solution on output

@code
    dg::DVec two(100,2), five(100,5), result(100, 12);
    dg::blas1::axpbypgz( 2.5, two, 2., five, -3.,result);
    //result[i] = -21 (2.5*2+2*5-3*12)
@endcode
 */
template< class container>
inline void axpbypgz( get_value_type<container> alpha, const container& x, get_value_type<container> beta, const container& y, get_value_type<container> gamma, container& z)
{
    dg::blas1::detail::doAxpbypgz( alpha, x, beta, y, gamma, z, get_vector_category<container>() );
    return;
}

/*! @brief \f$ y = f(y , g(x))\f$
 *
 * This routine elementwise evaluates \f[ f(y_i, g(x_{i})) \f]
 * This is strictly speaking not a BLAS routine since f and g can be a nonlinear functions.
 * @copydoc hide_container
 * @tparam BinarySubroutine Functor with signature: \c void \c operator()( value_type&, value_type) i.e. it writes into the first and reads from its second argument
 * @tparam UnaryOp Functor with signature: \c value_type \c operator()( value_type)
 * @param y contains result
 * @param f The subroutine
 * @param g unary operator
 * @param x input
 * @note the Functors must be callable on the device in use. In particular, with CUDA their signatures must contain the \__device__ specifier (s.a. \ref DG_DEVICE)
 * @note all aliases allowed
@code
    dg::HVec pi(20, M_PI), result(20, 0);
    dg::blas1::evaluate( result, dg::equals(), cos, pi);
    //result[i] =  -1. (cos(M_PI))
@endcode
 */
template< class container, class BinarySubroutine, class UnaryOp>
inline void evaluate( container& y, BinarySubroutine f, UnaryOp g, const container& x)
{
    dg::blas1::detail::doEvaluate(  get_vector_category<container>(), y, f, g, x);
    return;
}

/*! @brief \f$ z = f(z, g(x,y)\f$
 *
 * This routine elementwise evaluates \f[ f(z_i , g(x_{i}, y_{i})) \f]
 * This is strictly speaking not a BLAS routine since f can be a nonlinear function.
 * @copydoc hide_container
 * @tparam BinarySubroutine Functor with signature: \c void \c operator()( value_type&, value_type) i.e. it writes into the first and reads from its second argument
 * @tparam BinaryOp Functor with signature: \c value_type \c operator()( value_type, value_type)
 * @param z contains result
 * @param f The subroutine
 * @param g unary operator
 * @param x input 1
 * @param y input 2
 * @note the Functor must be callable on the device in use. In particular, with CUDA its signature must contain the \__device__ specifier. (s.a. \ref DG_DEVICE)
 * @note all aliases allowed
@code
    double function( double x, double y) {return sin(x)*sin(y);}
    dg::HVec pi2(20, M_PI/2.), pi3( 20, 3*M_PI/2.), result(20, 0);
    dg::blas1::evaluate( result, dg::equals(), function, pi2, pi3);
    //result[i] =  -1. (sin(M_PI/2.)*sin(3*M_PI/2.))
@endcode
 */
template< class container, class BinarySubroutine, class BinaryOp>
inline void evaluate( container& z, BinarySubroutine f, BinaryOp g, const container& x, const container& y)
{
    dg::blas1::detail::doEvaluate( get_vector_category<container>(), z, f, g, x, y);
    return;
}

/*! @brief \f$ y = op(x)\f$
 *
 * This routine computes \f[ y_i = op(x_i) \f]
 * This is strictly speaking not a BLAS routine since f can be a nonlinear function.
 * @copydoc hide_container
 * @tparam UnaryOp Functor with signature: \c value_type \c operator()( value_type)
 * @param x container x may alias y
 * @param y container y contains result, may alias x
 * @param op unary Operator to use on every element
 * @note the Functor must be callable on the device in use. In particular, with CUDA its signature must contain the \__device__ specifier. (s.a. \ref DG_DEVICE)

@code
    dg::DVec two( 100,2), result(100);
    dg::blas1::transform( two, result, dg::EXP<double>());
    //result[i] = 7.389056... (e^2)
@endcode
 */
template< class container, class UnaryOp>
inline void transform( const container& x, container& y, UnaryOp op )
{
    dg::blas1::evaluate( y, dg::equals(), op, x);
}

/*! @brief \f$ x = \alpha x\f$
 *
 * This routine computes \f[ \alpha x_i \f]
 * @copydoc hide_container
 * @param alpha Scalar
 * @param x container x

@code
    dg::DVec two( 100,2);
    dg::blas1::scal( two,  0.5 )); //result[i] = 1.
@endcode
 */
template< class container>
inline void scal( container& x, get_value_type<container> alpha)
{
    dg::blas1::detail::doScal( x, alpha, get_vector_category<container>() );
    return;
}

/*! @brief \f$ x = x + \alpha \f$
 *
 * This routine computes \f[ x_i + \alpha \f]
 * @copydoc hide_container
 * @param alpha Scalar
 * @param x container x

@code
    dg::DVec two( 100,2);
    dg::blas1::plus( two,  2. )); //result[i] = 4.
@endcode
 */
template< class container>
inline void plus( container& x, get_value_type<container> alpha)
{
    dg::blas1::detail::doPlus( x, alpha, get_vector_category<container>() );
    return;
}

/*! @brief \f$ y = x_1 x_2 \f$
*
* Multiplies two vectors element by element: \f[ y_i = x_{1i}x_{2i}\f]  i iterates over @b all elements inside the container. If \c container has the \c VectorVectorTag, i recursively loops over all entries. If the container sizes
 * do not match, the result is undefined.

* @copydoc hide_container
* @param x1 container x1
* @param x2 container x2 may alias x1
* @param y  container y contains result on output ( may alias x1 or x2)

@code
    dg::DVec two( 100,2), three( 100,3), result(100);
    dg::blas1::pointwiseDot( two,  three, result ); //result[i] = 6.
@endcode
*/
template< class container>
inline void pointwiseDot( const container& x1, const container& x2, container& y)
{
    dg::blas1::detail::doPointwiseDot( 1, x1, x2, 0, y, get_vector_category<container>() );
    return;
}

/**
* @brief \f$ y = \alpha x_1 x_2 + \beta y\f$
*
* Multiplies two vectors element by element: \f[ y_i = \alpha x_{1i}x_{2i} + \beta y_i\f]  i iterates over @b all elements inside the container. If \c container has the \c VectorVectorTag, i recursively loops over all entries. If the container sizes
 * do not match, the result is undefined.

* @copydoc hide_container
* @param alpha scalar
* @param x1 container x1
* @param x2 container x2 may alias x1
* @param beta scalar
* @param y  container y contains result on output ( may alias x1 or x2)

@code
    dg::DVec two( 100,2), three( 100,3), result(100,6);
    dg::blas1::pointwiseDot(2., two,  three, -4., result );
    //result[i] = -12. (2*2*3-4*6)
@endcode
*/
template< class container>
inline void pointwiseDot( get_value_type<container> alpha, const container& x1, const container& x2, get_value_type<container> beta, container& y)
{
    dg::blas1::detail::doPointwiseDot( alpha, x1, x2, beta, y, get_vector_category<container>() );
}

/**
* @brief \f$ y = \alpha x_1 x_2 x_3 + \beta y\f$
*
* Multiplies three vectors element by element: \f[ y_i = \alpha x_{1i}x_{2i}x_{3i} + \beta y_i\f]  i iterates over @b all elements inside the container. If \c container has the \c VectorVectorTag, i recursively loops over all entries. If the container sizes
 * do not match, the result is undefined.

* @copydoc hide_container
* @param alpha scalar
* @param x1 container x1
* @param x2 container x2 may alias x1
* @param x3 container x3 may alias x1 and/or x2
* @param beta scalar
* @param y  container y contains result on output ( may alias x1,x2 or x3)

@code
    dg::DVec two( 100,2), three( 100,3), four(100,4), result(100,6);
    dg::blas1::pointwiseDot(2., two,  three, four, -4., result );
    //result[i] = 24. (2*2*3*4-4*6)
@endcode
*/
template< class container>
inline void pointwiseDot( get_value_type<container> alpha, const container& x1, const container& x2, const container& x3, get_value_type<container> beta, container& y)
{
    dg::blas1::detail::doPointwiseDot( alpha, x1, x2, x3, beta, y, get_vector_category<container>() );
}

/**
* @brief \f$ y = x_1/ x_2\f$
*
* Divides two vectors element by element: \f[ y_i = x_{1i}/x_{2i}\f]  i iterates over @b all elements inside the container. If \c container has the \c VectorVectorTag, i recursively loops over all entries. If the container sizes
 * do not match, the result is undefined.

* @copydoc hide_container
* @param x1 container x1
* @param x2 container x2 may alias x1
* @param y  container y contains result on output ( may alias x1 and/or x2)

@code
    dg::DVec two( 100,2), three( 100,3), result(100);
    dg::blas1::pointwiseDivide( two,  three, result );
    //result[i] = -0.666... (2/3)
@endcode
*/
template< class container>
inline void pointwiseDivide( const container& x1, const container& x2, container& y)
{
    dg::blas1::detail::doPointwiseDivide( 1, x1, x2, 0, y, get_vector_category<container>() );
    return;
}
/**
* @brief \f$ y = \alpha x_1/ x_2 + \beta y \f$
*
* Divides two vectors element by element: \f[ y_i = \alpha x_{1i}/x_{2i} + \beta y_i \f]  i iterates over @b all elements inside the container. If \c container has the \c VectorVectorTag, i recursively loops over all entries. If the container sizes
 * do not match, the result is undefined.

* @copydoc hide_container
* @param alpha scalar
* @param x1 container x1
* @param x2 container x2 may alias x1
* @param beta scalar
* @param y  container y contains result on output ( may alias x1 and/or x2)

@code
    dg::DVec two( 100,2), three( 100,3), result(100,1);
    dg::blas1::pointwiseDivide( 3, two,  three, 5, result );
    //result[i] = 7 (3*2/3+5*1)
@endcode
*/
template< class container>
inline void pointwiseDivide( get_value_type<container> alpha, const container& x1, const container& x2, get_value_type<container> beta, container& y)
{
    dg::blas1::detail::doPointwiseDivide( alpha, x1, x2, beta, y, get_vector_category<container>() );
}

/**
* @brief \f$ z = \alpha x_1x_2 + \beta x_2y_2 + \gamma z\f$
*
* Multiplies and adds vectors element by element: \f[ z_i = \alpha x_{1i}y_{1i} + \beta x_{2i}y_{2i} + \gamma z_i \f]  i iterates over @b all elements inside the container. If \c container has the \c VectorVectorTag, i recursively loops over all entries. If the container sizes
 * do not match, the result is undefined.
* @copydoc hide_container
* @param alpha scalar
* @param x1 container x1
* @param y1 container y1
* @param beta scalar
* @param x2 container x2
* @param y2 container y2
* @param gamma scalar
* @param z  container z contains result on output
* @note all aliases are allowed

@code
    dg::DVec two(100,2), three(100,3), four(100,5), five(100,5), result(100,6);
    dg::blas1::pointwiseDot(2., two,  three, -4., four, five, 2., result );
    //result[i] = -56.
@endcode
*/
template<class container>
void pointwiseDot(  get_value_type<container> alpha, const container& x1, const container& y1,
                    get_value_type<container> beta,  const container& x2, const container& y2,
                    get_value_type<container> gamma, container & z)
{
    dg::blas1::detail::doPointwiseDot( alpha, x1, y1, beta, x2, y2, gamma, z, get_vector_category<container>() );
    return;
}
///@}
}//namespace blas1

///@cond
//forwarding function calls
template< class T1, class T2>
void transfer(T1&& arg1, T2&& arg2){
    blas1::transfer( std::forward<T1>(arg1), std::forward<T2>(arg2));
}
template< class T1, class T2>
T1 transfer(T2&& arg){
    return blas1::transfer<T1,T2>( std::forward<T2>(arg));
}

///@endcond


} //namespace dg

