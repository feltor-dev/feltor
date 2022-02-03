#pragma once
#include "dg/topology/functions.h"
#include "dg/backend/config.h"

namespace dg{
/**
 * @brief \f$ f(x) = x\f$
 * @ingroup basics
 */
struct IDENTITY
{
    template<class T>
    DG_DEVICE T operator()(T x)const{return x;}
};


///@addtogroup binary_operators
///@{

///\f$ y=x\f$
struct equals
{
    template< class T1, class T2>
DG_DEVICE void operator()( T1 x, T2& y) const
    {
        y = x;
    }
};
///\f$ y=y+x\f$
struct plus_equals
{
    template< class T1, class T2>
DG_DEVICE void operator()( T1 x, T2& y) const
    {
        y += x;
    }
};
///\f$ y=y-x\f$
struct minus_equals
{
    template< class T1, class T2>
DG_DEVICE void operator()( T1 x, T2& y) const
    {
        y -= x;
    }
};
///\f$ y=xy\f$
struct times_equals
{
    template< class T1, class T2>
DG_DEVICE void operator()( T1 x, T2& y) const
    {
        y *= x;
    }
};
///\f$ y = y/x\f$
struct divides_equals
{
    template< class T1, class T2>
DG_DEVICE void operator()( T1 x, T2& y) const
    {
        y /= x;
    }
};
///@}

///@addtogroup variadic_evaluates
///@{

///\f$ y = x_1/x_2 \f$
struct divides
{
    template< class T1, class T2>
DG_DEVICE T1 operator()( T1 x1, T2 x2) const
    {
        return x1/x2;
    }
};

///@brief \f$ y = \sum_i x_i \f$
struct Sum
{
    ///@brief \f[ \sum_i x_i \f]
    template< class T1, class ...Ts>
DG_DEVICE T1 operator()( T1 x, Ts... rest) const
    {
        T1 tmp = T1{0};
        sum( tmp, x, rest...);
        return tmp;
    }
    private:
    template<class T, class ...Ts>
DG_DEVICE void sum( T& tmp, T x, Ts... rest) const
    {
        tmp += x;
        sum( tmp, rest...);
    }

    template<class T>
DG_DEVICE void sum( T& tmp, T x) const
    {
        tmp += x;
    }
};

///@brief \f$ y = \sum_i a_i x_i \f$
struct PairSum
{
    ///@brief \f[ \sum_i a_i x_i \f]
    template< class T, class ...Ts>
DG_DEVICE T operator()( T a, T x, Ts... rest) const
    {
        T tmp = T{0};
        sum( tmp, a, x, rest...);
        return tmp;
    }
    private:
    template<class T, class ...Ts>
DG_DEVICE void sum( T& tmp, T alpha, T x, Ts... rest) const
    {
        tmp = DG_FMA( alpha, x, tmp);
        sum( tmp, rest...);
    }

    template<class T>
DG_DEVICE void sum( T& tmp, T alpha, T x) const
    {
        tmp = DG_FMA(alpha, x, tmp);
    }
};
///@brief \f$ y = \sum_i a_i x_i y_i \f$
struct TripletSum
{
    ///@brief \f[ \sum_i \alpha_i x_i y_i \f]
    template< class T1, class ...Ts>
DG_DEVICE T1 operator()( T1 a, T1 x1, T1 y1, Ts... rest) const
    {
        T1 tmp = T1{0};
        sum( tmp, a, x1, y1, rest...);
        return tmp;
    }
    private:
    template<class T, class ...Ts>
DG_DEVICE void sum( T& tmp, T alpha, T x, T y, Ts... rest) const
    {
        tmp = DG_FMA( alpha*x, y, tmp);
        sum( tmp, rest...);
    }

    template<class T>
DG_DEVICE void sum( T& tmp, T alpha, T x, T y) const
    {
        tmp = DG_FMA(alpha*x, y, tmp);
    }
};

///@}

///@addtogroup variadic_subroutines
///@{

///@brief \f$ y = \sum_i a_i x_i + b y,\quad \tilde y = \sum_i \tilde a_i x_i + \tilde b y \f$
struct EmbeddedPairSum
{
    ///@brief \f[ \sum_i \alpha_i x_i \f]
    template< class T1, class ...Ts>
DG_DEVICE void operator()( T1& y, T1& yt, T1 b, T1 bt, Ts... rest) const
    {
        y = b*y;
        yt = bt*yt;
        sum( y, yt, rest...);
    }
    private:
    template< class T1,  class ...Ts>
DG_DEVICE void sum( T1& y_1, T1& yt_1, T1 b, T1 bt, T1 k, Ts... rest) const
    {
        y_1 = DG_FMA( b, k, y_1);
        yt_1 = DG_FMA( bt, k, yt_1);
        sum( y_1, yt_1, rest...);
    }

    template< class T1>
DG_DEVICE void sum( T1& y_1, T1& yt_1, T1 b, T1 bt, T1 k) const
    {
        y_1 = DG_FMA( b, k, y_1);
        yt_1 = DG_FMA( bt, k, yt_1);
    }
};

/// \f$ f( y, g(x_0, ..., x_s)) \f$
template<class BinarySub, class Functor>
struct Evaluate
{
    Evaluate( BinarySub sub, Functor g):
        m_f( sub),
        m_g( g) {}
#ifdef __CUDACC__
// cuda compiler spits out a lot of warnings if
// e.g. dg::transform is used on host vectors with host function
// hd_warning_disable is unfortunately undocumented, but let's try
// If it ever causes trouble we can remove it again
// it just suppresses compiler warnings:
// https://stackoverflow.com/questions/55481202/how-to-disable-cuda-host-device-warning-for-just-one-function
#pragma hd_warning_disable
#endif
    template< class T, class... Ts>
DG_DEVICE void operator() ( T& y, Ts... xs){
        m_f(m_g(xs...), y);
    }
    private:
    BinarySub m_f;
    Functor m_g;
};

/// \f$ y\leftarrow ay \f$
template<class T>
struct Scal
{
    Scal( T a): m_a(a){}
DG_DEVICE
    void operator()( T& y)const{
        y *= m_a;
    }
    private:
    T m_a;
};

/// \f$ y\leftarrow y+a \f$
template<class T>
struct Plus
{
    Plus( T a): m_a(a){}
DG_DEVICE
    void operator()( T& y) const{
        y += m_a;
    }
    private:
    T m_a;
};

///@brief \f$ y\leftarrow ax+by \f$
///@ingroup binary_operators
template<class T>
struct Axpby
{
    Axpby( T a, T b): m_a(a), m_b(b){}
DG_DEVICE
    void operator()( T x, T& y)const {
        T temp = y*m_b;
        y = DG_FMA( m_a, x, temp);
    }
    private:
    T m_a, m_b;
};
///@brief \f$ y\leftarrow axy+by \f$
///@ingroup binary_operators
template<class T>
struct AxyPby
{
    AxyPby( T a, T b): m_a(a), m_b(b){}
DG_DEVICE
    void operator()( T x, T& y)const {
        T temp = y*m_b;
        y = DG_FMA( m_a*x, y, temp);
    }
    private:
    T m_a, m_b;
};

/// \f$ z\leftarrow ax+by+gz \f$
template<class T>
struct Axpbypgz
{
    Axpbypgz( T a, T b, T g): m_a(a), m_b(b), m_g(g){}
DG_DEVICE
    void operator()( T x, T y, T& z)const{
        T temp = z*m_g;
        temp = DG_FMA( m_a, x, temp);
        temp = DG_FMA( m_b, y, temp);
        z = temp;
    }
    private:
    T m_a, m_b, m_g;
};

/// \f$ z\leftarrow ax_1y_1+bx_2y_2+gz \f$
template<class T>
struct PointwiseDot
{
    PointwiseDot( T a, T b, T g = (T)0): m_a(a), m_b(b), m_g(g) {}
    ///\f$ z = axy+bz \f$
DG_DEVICE void operator()( T x, T y, T& z)const{
        T temp = z*m_b;
        z = DG_FMA( m_a*x, y, temp);
    }
    ///\f$ y = ax_1x_2x_3 +by \f$
DG_DEVICE
    void operator()( T x1, T x2, T x3, T& y)const{
        T temp = y*m_b;
        y = DG_FMA( m_a*x1, x2*x3, temp);
    }
    /// \f$ z = ax_1y_1+bx_2y_2+gz \f$
DG_DEVICE
    void operator()( T x1, T y1, T x2, T y2, T& z)const{
        T temp = z*m_g;
        temp = DG_FMA( m_a*x1, y1, temp);
        temp = DG_FMA( m_b*x2, y2, temp);
        z = temp;
    }
    private:
    T m_a, m_b, m_g;
};

/// \f$ z\leftarrow ax/y + bz \f$
template<class T>
struct PointwiseDivide
{
    PointwiseDivide( T a, T b): m_a(a), m_b(b){}
    ///\f$ z = az/y +bz \f$
DG_DEVICE
    void operator()( T y, T& z)const{
        T temp = z*m_b;
        z = DG_FMA( m_a, z/y, temp);
    }
DG_DEVICE
    void operator()( T x, T y, T& z)const{
        T temp = z*m_b;
        z = DG_FMA( m_a, x/y, temp);
    }
    private:
    T m_a, m_b;
};
///@}

///@cond
namespace detail
{
template<class F, class G>
struct Compose
{
    Compose( F f, G g):m_f(f), m_g(g){}
    template<class ...Xs>
    auto operator() ( Xs&& ... xs){
        return m_f(m_g(std::forward<Xs>(xs)...));
    }
    template<class ...Xs>
    auto operator() ( Xs&& ... xs) const {
        return m_f(m_g(std::forward<Xs>(xs)...));
    }
    private:
    F m_f;
    G m_g;
};
}//namespace detail
///@endcond

///@addtogroup composition
///@{
/**
 * @brief Create Composition functor \f$ f(g(x_0,x_1,...)) \f$
 *
 * @code{.cpp}
 * dg::Grid2d grid2d( -1., 1., -1., 1., 3, 40, 40);
 * //Mark everything above 2 with 1s and below with 0s
 * dg::HVec fg = dg::evaluate( dg::compose( dg::Heaviside( 2.), dg::Gaussian( 0., 0., 2., 2., 4.)), grid2d);
 * @endcode
 * @tparam UnaryOp Model of Unary Function taking the return type of g \c return_type_f \c f(return_type_g)
 * @tparam Functor Inner functor, takes an arbitrary number of parameters and returns one \c return_type_g \c g(value_type0,value_type1,...)
 * @param f outer functor
 * @param g inner functor
 * @attention only works for host functions. The rationale is that this
 * function is intended to work with lambda functions and previously nvcc did
 * not support lambdas. If a version for device functions is ever needed it can
 * be provided.
 *
 * @return a function object that forwards all parameters to g and returns the
 * return value of f, which is \f$ f(g(x_0,x_1,...)) \f$
 */
template <class UnaryOp, class Functor>
auto compose( UnaryOp f, Functor g) {
    return detail::Compose<UnaryOp,Functor>( f, g);
    //a C++-14 way of generating a generic lambda with a parameter pack. Taken from:
    //https://stackoverflow.com/questions/19071268/function-composition-in-c-c11
    //return [f,g](auto&&... xs){ return f(g(std::forward<decltype(xs)>(xs)...));};
}
/**@brief Create Composition funtor of an arbitrary number of functions \f$ f_0(f_1(f_2( ... f_s(x_0, x_1, ...)))\f$
 *
 * @tparam UnaryOp Model of Unary Function taking the return type of f_1 \c return_type_f0 \c f0(return_type_f1)
 * @tparam Fs UnaryOps except the innermost functor, which takes an arbitrary number of parameters and returns one \c return_type_fs \c f_s(value_type0,value_type1,...)
 */
template <class UnaryOp, typename... Functors>
auto compose(UnaryOp f0, Functors... fs) {
    return compose( f0 , compose(fs...));
}
///@}

}//namespace dg
