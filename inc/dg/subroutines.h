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
DG_DEVICE auto operator()( T1 x1, T2 x2) const
    {
        return x1/x2;
    }
};

///@brief \f$ y = \sum_i x_i \f$
struct Sum
{
    ///@brief \f[ \sum_i x_i \f]
    template< class T1, class ...Ts>
DG_DEVICE auto operator()( T1 x, Ts... rest) const
    {
        // unfortunately the fold expression ( x + ... + rest)
        // does currently not guarantee the order of execution
        // so we need to wait for DR 2611 to be implemented in g++ to use it
        return sum( x, rest ...);
    }
    private:
    template< class T1, class ...Ts>
DG_DEVICE auto sum( T1 x, Ts... rest) const
    {
        return x + sum( rest...);
    }

    template<class T1>
DG_DEVICE auto sum( T1 x1) const
    {
        return x1;
    }
};


///@brief \f$ y = \prod_i x_i \f$
struct Product
{
    ///@brief \f[ \sum_i x_i \f]
    template< class T1, class ...Ts>
DG_DEVICE auto operator()( T1 x, Ts... rest) const
    {
        // manual implement ( x * ... * rest) until DR 2611 is resolved
        return prod(x, rest...);
    }
    private:
    template< class T1, class ...Ts>
DG_DEVICE auto prod( T1 x, Ts... rest) const
    {
        return x * prod( rest...);
    }

    template<class T1>
DG_DEVICE auto prod( T1 x1) const
    {
        return x1;
    }
};

///@brief \f$ y = \sum_i a_i x_i \f$
struct PairSum
{
    ///@brief \f[ \sum_i a_i x_i \f]
    template< class T1, class T2, class ...Ts>
DG_DEVICE auto operator()( T1 a, T2 x, Ts... rest) const
    {
        return sum( a, x, rest...);
    }
    private:
    template<class T1, class T2, class ...Ts>
DG_DEVICE auto sum( T1 alpha, T2 x, Ts... rest) const
    {
        return DG_FMA( alpha, x, sum(rest...));
    }

    template< class T1, class T2>
DG_DEVICE auto sum( T1 alpha, T2 x) const
    {
        return alpha*x;
    }
};
///@brief \f$ y = \sum_i a_i x_i y_i \f$
struct TripletSum
{
    ///@brief \f[ \sum_i \alpha_i x_i y_i \f]
    template< class T0, class T1, class T2, class ...Ts>
DG_DEVICE auto operator()( T0 a, T1 x1, T2 y1, Ts... rest) const
    {
        return sum( a, x1, y1, rest...);
    }
    private:
    template<class T0, class T1, class T2, class ...Ts>
DG_DEVICE auto sum( T0 alpha, T1 x, T2 y, Ts... rest) const
    {
        return DG_FMA( alpha*x, y, sum(rest...));
    }

    template<class T0, class T1, class T2>
DG_DEVICE auto sum( T0 alpha, T1 x, T2 y) const
    {
        return (alpha*x)*y;
    }
};

///@}

///@addtogroup variadic_subroutines
///@{

///@brief \f$ y = \sum_i b_i x_i + b_0 y,\quad \tilde y = \sum_i \tilde b_i x_i + \tilde b_0 y \f$
struct EmbeddedPairSum
{
    ///@brief \f$ y = \sum_i b_i x_i + b_0 y,\quad \tilde y = \sum_i \tilde b_i x_i + \tilde b_0 y \f$
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
    Evaluate( BinarySub sub, Functor g): m_f( sub), m_g( g) {}
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
    template<class T1>
DG_DEVICE
    void operator()( T1& y)const{
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
    template<class T1>
DG_DEVICE
    void operator()( T1& y) const{
        y += m_a;
    }
    private:
    T m_a;
};

///@brief \f$ y\leftarrow ax+by \f$
///@ingroup binary_operators
template<class T0, class T1>
struct Axpby
{
    Axpby( T0 a, T1 b): m_a(a), m_b(b){}
    template<class T2, class T3>
DG_DEVICE
    void operator()( T2 x, T3& y)const {
        y *= m_b;
        y = DG_FMA( m_a, x, y);
    }
    private:
    T0 m_a;
    T1 m_b;
};
///@brief \f$ y\leftarrow axy+by \f$
///@ingroup binary_operators
template<class T0, class T1>
struct AxyPby
{
    AxyPby( T0 a, T1 b): m_a(a), m_b(b){}
    template<class T2, class T3>
DG_DEVICE
    void operator()( T2 x, T3& y)const {
        y *= m_b;
        y = DG_FMA( m_a*x, y, y);
    }
    private:
    T0 m_a;
    T1 m_b;
};

/// \f$ z\leftarrow ax+by+gz \f$
template<class T0, class T1, class T2>
struct Axpbypgz
{
    Axpbypgz( T0 a, T1 b, T2 g): m_a(a), m_b(b), m_g(g){}
    template<class T3, class T4, class T5>
DG_DEVICE
    void operator()( T3 x, T4 y, T5& z)const{
        z *= m_g;
        z = DG_FMA( m_a, x, z);
        z = DG_FMA( m_b, y, z);
    }
    private:
    T0 m_a;
    T1 m_b;
    T2 m_g;
};

/// \f$ z\leftarrow ax_1y_1+bz \f$
template<class T0, class T1>
struct PointwiseDot
{
    PointwiseDot( T0 a, T1 b): m_a(a), m_b(b) {}
    template<class T3, class T4, class T5>
    ///\f$ z = axy+bz \f$
DG_DEVICE void operator()( T3 x, T4 y, T5& z)const{
        z *= m_b;
        z = DG_FMA( m_a*x, y, z);
    }
    ///\f$ y = ax_1x_2x_3 +by \f$
    template<class T3, class T4, class T5, class T6>
DG_DEVICE
    void operator()( T3 x1, T4 x2, T5 x3, T6& y)const{
        y *= m_b;
        y = DG_FMA( m_a*x1, x2*x3, y);
    }
    private:
    T0 m_a;
    T1 m_b;
};
/// \f$ z\leftarrow ax_1y_1+bx_2y_2+gz \f$
template<class T0, class T1, class T2>
struct PointwiseDot2
{
    PointwiseDot2( T0 a, T1 b, T2 g): m_a(a), m_b(b), m_g(g) {}
    /// \f$ z = ax_1y_1+bx_2y_2+gz \f$
    template<class T3, class T4, class T5, class T6, class T7>
DG_DEVICE
    void operator()( T3 x1, T4 y1, T5 x2, T6 y2, T7& z)const{
        z *= m_g;
        z = DG_FMA( m_a*x1, y1, z);
        z = DG_FMA( m_b*x2, y2, z);
    }
    private:
    T0 m_a;
    T1 m_b;
    T2 m_g;
};

/// \f$ z\leftarrow ax/y + bz \f$
template<class T0, class T1>
struct PointwiseDivide
{
    PointwiseDivide( T0 a, T1 b): m_a(a), m_b(b){}
    ///\f$ z = az/y +bz \f$
    template<class T3, class T4>
DG_DEVICE
    void operator()( T3 y, T4& z)const{
        z *= m_b;
        z = DG_FMA( m_a, z/y, z);
    }
    template<class T3, class T4, class T5>
DG_DEVICE
    void operator()( T3 x, T4 y, T5& z)const{
        z *= m_b;
        z = DG_FMA( m_a, x/y, z);
    }
    private:
    T0 m_a;
    T1 m_b;
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
