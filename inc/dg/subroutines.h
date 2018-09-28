#pragma once
#include "dg/geometry/functions.h"
#include "dg/backend/config.h"

namespace dg{

///@addtogroup functions
///@{

///\f$ x=y\f$
struct equals
{
    template< class T1, class T2>
DG_DEVICE void operator()( T1& out, T2 in) const
    {
        out = in;
    }
};
///\f$ x+=y\f$
struct plus_equals
{
    template< class T1, class T2>
DG_DEVICE void operator()( T1& out, T2 in) const
    {
        out += in;
    }
};
///\f$ x-=y\f$
struct minus_equals
{
    template< class T1, class T2>
DG_DEVICE void operator()( T1& out, T2 in) const
    {
        out -= in;
    }
};
///\f$ x*=y\f$
struct times_equals
{
    template< class T1, class T2>
DG_DEVICE void operator()( T1& out, T2 in) const
    {
        out *= in;
    }
};
///\f$ x/=y\f$
struct divides_equals
{
    template< class T1, class T2>
DG_DEVICE void operator()( T1& out, T2 in) const
    {
        out /= in;
    }
};

///@brief \f[ \sum_i x_i \f]
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
    template<class T, class T2, class ...Ts>
DG_DEVICE void sum( T& tmp, T2 x, Ts... rest) const
    {
        tmp += x;
        sum( tmp, rest...);
    }

    template<class T, class T2>
DG_DEVICE void sum( T& tmp, T2 x) const
    {
        tmp += x;
    }
};

///@brief \f[ \sum_i \alpha_i x_i \f]
struct PairSum
{
    ///@brief \f[ \sum_i \alpha_i x_i \f]
    template< class T1, class T2, class ...Ts>
DG_DEVICE T1 operator()( T1 alpha, T2 x, Ts... rest) const
    {
        T1 tmp = T1{0};
        sum( tmp, alpha, x, rest...);
        return tmp;
    }
    private:
    template<class T, class T1, class T2, class ...Ts>
DG_DEVICE void sum( T& tmp, T1 alpha, T2 x, Ts... rest) const
    {
        tmp = DG_FMA( alpha, x, tmp);
        sum( tmp, rest...);
    }

    template<class T, class T1, class T2>
DG_DEVICE void sum( T& tmp, T1 alpha, T2 x) const
    {
        tmp = DG_FMA(alpha, x, tmp);
    }
};
///@brief \f[ \sum_i \alpha_i x_i y_i \f]
struct TripletSum
{
    ///@brief \f[ \sum_i \alpha_i x_i y_i \f]
    template< class T1, class T2, class T3, class ...Ts>
DG_DEVICE T1 operator()( T1 alpha, T2 x1, T3 y1, Ts... rest) const
    {
        T1 tmp = T1{0};
        sum( tmp, alpha, x1, y1, rest...);
        return tmp;
    }
    private:
    template<class T, class T1, class T2, class T3, class ...Ts>
DG_DEVICE void sum( T& tmp, T1 alpha, T2 x, T3 y, Ts... rest) const
    {
        tmp = DG_FMA( alpha*x, y, tmp);
        sum( tmp, rest...);
    }

    template<class T, class T1, class T2, class T3>
DG_DEVICE void sum( T& tmp, T1 alpha, T2 x, T3 y) const
    {
        tmp = DG_FMA(alpha*x, y, tmp);
    }
};

///@}
///@cond
template<class BinarySub, class Functor>
struct Evaluate
{
    Evaluate( BinarySub sub, Functor g):
        m_f( sub),
        m_g( g) {}
    template< class T, class... Ts>
DG_DEVICE void operator() ( T& y, Ts... xs){
        m_f(y, m_g(xs...));
    }
    private:
    BinarySub m_f;
    Functor m_g;
};

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

template<class T>
struct PointwiseDot
{
    PointwiseDot( T a, T b, T g = (T)0): m_a(a), m_b(b), m_g(g) {}
DG_DEVICE
    void operator()( T x, T &y)const{
        T temp = y*m_b;
        y = DG_FMA( m_a*x, y, temp);
    }
DG_DEVICE
    void operator()( T x, T y, T& z)const{
        T temp = z*m_b;
        z = DG_FMA( m_a*x, y, temp);
    }
DG_DEVICE
    void operator()( T x1, T x2, T x3, T& y)const{
        T temp = y*m_b;
        y = DG_FMA( m_a*x1, x2*x3, temp);
    }
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

template<class T>
struct PointwiseDivide
{
    PointwiseDivide( T a, T b): m_a(a), m_b(b){}
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
///@endcond


}//namespace dg
