#pragma once

#include <cmath>
//! M_PI is non-standard ... so MSVC complains
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <vector>
#include <random>
#include <functional>
#include "blas1.h"
#include "topology/grid.h"
#include "topology/evaluation.h"
#include "topology/functions.h"
/*!@file
 * Functors to use in dg::evaluate or dg::blas1::transform functions
 */
namespace dg
{

///@addtogroup basics
///@{
//Everything that is quite basic and simple


///@brief \f$ f(x) = x + c\f$
template <class T = double>
struct PLUS
{
    /**
     * @brief Construct
     *
     * @param value the constant c to be added
     */
    PLUS( T value): x_(value){}
    DG_DEVICE
    T operator()( T x)const{ return x + x_;}
    private:
    T x_;
};

///@brief \f$ f(x) = \exp( x)\f$
template< class T = double >
struct EXP
{
    DG_DEVICE T operator() ( T x) const
    {
        return exp(x);
    }
};

///@brief \f$ f(x) = \ln(x)\f$
template < class T = double>
struct LN
{
    DG_DEVICE T operator() (const T& x) const
    {
        return log(x);
    }

};

///@brief \f$ f(x) = \sqrt{x}\f$
template < class T = double>
struct SQRT
{
    DG_DEVICE T operator() (T x) const
    {
        return sqrt(x);
    }
};

///@brief \f$ f(x) = x^2\f$
struct Square
{
    template<class T>
    DG_DEVICE T operator()( T x) const{ return x*x;}
};

///@brief \f$ f(x) = \frac{1}{\sqrt{x}}\f$
template < class T = double>
struct InvSqrt
{
    DG_DEVICE T operator() (T x) const
    {
        return 1./sqrt(x);
    }
};

///@brief \f$ f(x) = 1/x \f$
template <class T = double>
struct INVERT
{
    DG_DEVICE T operator()( T x)const{ return 1./x;}
};

///@brief \f$ f(x) = |x|\f$
template <class T = double>
struct ABS
{
    DG_DEVICE T operator()(T x)const{ return fabs(x);}
};

/**
 * @brief
 * \f$ f(x) = \text{sgn}(x) = \begin{cases}
 *  -1 \text{ for } x < 0 \\
 *  0  \text{ for } x = 0 \\
 *  +1 \text{ for } x > 0
 *  \end{cases}\f$
 */
template <class T = double>
struct Sign
{
    DG_DEVICE T operator()(T x)const{ return (T(0) < x) - (x < T(0));}
};

///@brief \f$ f(x,y) = \max(|x|,|y|)\f$
template <class T = double>
struct AbsMax
{
    DG_DEVICE T operator() ( T x, T y) const
    {
        T absx = x>0 ? x : -x;
        T absy = y>0 ? y : -y;
        return absx > absy ? absx : absy;
    }
};

///@brief \f$ f(x,y) = \min(|x|,|y|)\f$
template <class T = double>
struct AbsMin
{
    DG_DEVICE T operator() (T x, T y) const
    {
        T absx = x<0 ? -x : x;
        T absy = y<0 ? -y : y;
        return absx < absy ? absx : absy;
    }
};

/**
 * @brief
 \f$ f(x) = \begin{cases}
         x \text{ for } x>0 \\
         0 \text{ else}
 \end{cases}
 \f$
 */
template <class T = double>
struct POSVALUE
{
    DG_DEVICE T operator()( T x)const{
        if (x >= 0.0) return x;
        return 0.0;
    }
};

/**
 * @brief \f$ f(x) = \f$ \c x mod m > 0 ? x mod m : x mod m + m
 *
 * returns (positive) modulo
 * @tparam T value type
 */
template <class T= double>
struct MOD
{
    /**
     * @brief Construct from modulo
     *
     * @param m modulo basis
     */
    MOD( T m): m_m(m){}

    DG_DEVICE
    T operator()( T x)const{
        return (fmod(x,m_m) < 0 ) ? (m_m + fmod(x,m_m)) : fmod(x,m_m);
    }
    private:
    T m_m;

};

/**
 * @brief \f$ f(x) = \mathrm{!std::isfinite(x)}\f$
 *
 * return true if \c x is \c NaN or \c Inf
@code
//Check if a vector contains Inf or NaN
thrust::device_vector<double> x( 100);
bool hasnan = dg::blas1::reduce( x, false, thrust::logical_or<bool>(),
    dg::ISNFINITE<double>());
std::cout << "x contains Inf or NaN "<<std::boolalpha<<hasnan<<"\n";
@endcode
 */
template <class T>
struct ISNFINITE
{
#ifdef __CUDACC__
    DG_DEVICE bool operator()(T x){ return !isfinite(x);}
#else
    bool operator()( T x){ return !std::isfinite(x);}
#endif
};

/**
 * @brief \f$ f(x) =\begin{cases} \mathrm{true\ if}\ |x| > 10^{100}\\
 * \mathrm{false\ else}
 * \end{cases}\f$
 *
 * Also return true if \c x is \c NaN or \c Inf.
 * The intention is to use this in the reduce function to debug code if
 * you get an error message of Inf or Nan from the \c dot function
@code
//Check if a vector contains is sane
thrust::device_vector<double> x( 100);
bool hasnan = dg::blas1::reduce( x, false, thrust::logical_or<bool>(),
    dg::ISNSANE<double>());
std::cout << "x contains insane numbers "<<std::boolalpha<<hasnan<<"\n";
@endcode
 */
template <class T>
struct ISNSANE
{
#ifdef __CUDACC__
    DG_DEVICE bool operator()(T x){
        if( !isfinite(x))
            return true;
        if( x > 1e100 || x < -1e100)
            return true;
        return false;
    }
#else
    bool operator()( T x){
        if( !std::isfinite(x))
            return true;
        if( x > 1e100 || x < -1e100)
            return true;
        return false;
    }
#endif
};

/**
 * @brief
 \f$ f(x_1, x_2, ...) = \begin{cases}
         \min(x_1, x_2, ...) &\text{ for } x_1, x_2, ... >0 \\
         \max(x_1, x_2, ...) &\text{ for } x_1, x_2, ... <0 \\
         0 &\text{ else}
 \end{cases}
 \f$
 *
 * Useful for Slope limiter
 */
struct MinMod
{
    ///@return minmod(x1, x2)
#ifdef __CUDACC__
    template < class T>
    DG_DEVICE T operator()( T x1, T x2) const
    {
        if( x1 > 0 && x2 > 0)
            return min(x1,x2);
        else if( x1 < 0 && x2 < 0)
            return max(x1,x2);
        return 0.;
    }
#else
    template < class T>
    T operator()( T x1, T x2) const
    {
        if( x1 > 0 && x2 > 0)
            return std::min(x1,x2);
        else if( x1 < 0 && x2 < 0)
            return std::max(x1,x2);
        return 0.;
    }
#endif
    ///@return minmod(x1, x2, x3);
    template<class T>
    DG_DEVICE T operator() ( T x1, T x2, T x3)const
    {
        return this-> operator()( this-> operator()( x1, x2), x3);
    }
};

/**
 * @brief \f$ f(x_1,x_2) = 2\begin{cases}
 *  \frac{x_1x_2}{x_1+x_2} &\text{ if } x_1x_2 > 0 \\
 *  0 & \text { else }
 *  \end{cases}
 *  \f$
 *  @note The first case is the harmonic mean between x_1 and x_2
 */
struct VanLeer
{
    template<class T>
    DG_DEVICE T operator()( T x1, T x2) const
    {
        if( x1*x2 <= 0)
            return 0.;
        return 2.*x1*x2/(x1+x2);
    }
};

/**
 * @brief \f$ \text{up}(v, b, f ) = \begin{cases}  b &\text{ if } v \geq 0 \\
 *  f &\text{ else}
 *  \end{cases}
 *  \f$
 */
struct Upwind
{
    template<class T>
    DG_DEVICE T operator()( T velocity, T backward, T forward) const{
        if( velocity >= 0)
            return backward;
        else
            return forward;
    }
};

/**
 * @brief \f$ \text{up}(v, b, f ) = v \begin{cases}  b &\text{ if } v \geq 0 \\
 *  f &\text{ else}
 *  \end{cases}
 *  \f$
 */
struct UpwindProduct
{
    template<class T>
    DG_DEVICE T operator()( T velocity, T backward, T forward)const{
        return velocity*m_up(velocity, backward, forward);
    }
    private:
    Upwind m_up;
};

/**
 * @brief \f$ \text{up}(v, g_m, g_0, g_p, h_m, h_p ) = \begin{cases}  +h_m \Lambda( g_0, g_m) &\text{ if } v \geq 0 \\
 *  -h_p \Lambda( g_p, g_0) &\text{ else}
 *  \end{cases}
 *  \f$
 *
 * @tparam Limiter Any two-dimensional functor
 * @sa VanLeer, MinMod
 */
template<class Limiter>
struct SlopeLimiter
{
    SlopeLimiter() {}
    SlopeLimiter( Limiter l ) : m_l( l){}
    template<class T>
    DG_DEVICE T operator()( T v, T gm, T g0, T gp, T hm, T hp ) const{
        if( v >= 0)
            return +hm*m_l( g0, gm);
        else
            return -hp*m_l( gp, g0);
    }
    private:
    Limiter m_l;
};

/**
 * @brief \f$ \text{up}(v, g_m, g_0, g_p, h_m, h_p ) = v \begin{cases}  +h_m \Lambda( g_0, g_m) &\text{ if } v \geq 0 \\
 *  -h_p \Lambda( g_p, g_0) &\text{ else}
 *  \end{cases}
 *  \f$
 *
 * @tparam Limiter Any two-dimensional functor
 * @sa VanLeer, MinMod
 */
template<class Limiter>
struct SlopeLimiterProduct
{
    SlopeLimiterProduct() {}
    SlopeLimiterProduct( Limiter l ) : m_s( l){}
    template<class T>
    DG_DEVICE T operator()( T v, T gm, T g0, T gp, T hm, T hp ) const{
        return v*m_s(v,gm,g0,gp,hm,hp);
    }
    private:
    SlopeLimiter<Limiter> m_s;
};
///@}


/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

///@addtogroup functions
///@{
//

/**
 * @brief
     \f$ f(\psi) = \begin{cases}
        1  \text{ if } \psi_{\min} < \psi < \psi_{\max}\\
        0  \text{ else}
     \end{cases}\f$
 */
struct Iris
{
    Iris( double psi_min, double psi_max ):
        m_psimin(psi_min), m_psimax(psi_max) { }
    DG_DEVICE
    double operator()(double psi)const
    {
        if( psi > m_psimax) return 0.;
        if( psi < m_psimin) return 0.;
        return 1.;
    }
    private:
    double m_psimin, m_psimax;
};
/**
 * @brief
     \f$ f(\psi) = \begin{cases}
        0  \text{ if } \psi > \psi_{\max} \\
        1  \text{ else}
     \end{cases}\f$
 */
struct Pupil
{
    Pupil( double psimax):
        psimax_(psimax) { }
    DG_DEVICE
    double operator()(double psi)const
    {
        if( psi > psimax_) return 0.;
        return 1.;
    }
    private:
    double psimax_;
};
/**
 * @brief
     \f$ f(\psi) = \begin{cases}
        \psi_{\max}  \text{ if } \psi > \psi_{\max} \\
        \psi \text{ else}
     \end{cases}\f$
 */
struct PsiPupil
{
    PsiPupil(double psimax):
        psimax_(psimax){ }
    DG_DEVICE
    double operator()(double psi)const
    {
        if( psi > psimax_) return psimax_;
        return  psi;
    }
    private:
    double psimax_;
};
/**
 * @brief
     \f$ f(x) = \begin{cases}
        0  \text{ if } x < x_b \\
        1  \text{ else}
     \end{cases}\f$
  @note the 1 is inclusive i.e if x==x_b the functor always returns 1
 */
struct Heaviside
{

    /**
     * @brief Construct with xb and sign
     *
     * @param xb boundary value
     * @param sign either +1 or -1, If -1, we mirror the Heaviside at
     *  the \c x=x_b axis, i.e. we swap the < sign in the definition to >
     * @note When sign is positive the function leaves the positive and damps the negative and vice versa when sign is negative the function leaves the negative and damps the positive.
     */
    Heaviside( double xb, int sign = +1):
        m_xb(xb), m_s(sign){ }

    DG_DEVICE
    double operator()(double x)const
    {
        if( (x < m_xb && m_s == 1) || (x > m_xb && m_s == -1)) return 0.;
        return 1.;
    }
    private:
    double m_xb;
    int m_s;
};


/**
 * @brief \f$ f(x,y) = \sqrt{ (x-x_0)^2 + (y-y_0)^2} \f$
 */
struct Distance
{
    Distance( double x0, double y0): m_x0(x0), m_y0(y0){}
    DG_DEVICE
    double operator()(double x, double y){
        return sqrt( (x-m_x0)*(x-m_x0) + (y-m_y0)*(y-m_y0));
    }
    private:
    double m_x0, m_y0;
};
/**
 * @brief
 * \f$ f(x) = y_1\frac{x-x_0}{x_1-x_0} + y_0\frac{x-x_1}{x_0-x_1}\f$
 *
 * The linear interpolation polynomial
 */
struct Line{
    Line(double x0, double y0, double x1, double y1) :
        m_x0(x0), m_y0(y0), m_x1(x1), m_y1(y1){}
    double operator()(double x){
        return m_y1*(x-m_x0)/(m_x1-m_x0) + m_y0*(x-m_x1)/(m_x0-m_x1);
    }
    private:
    double m_x0, m_y0, m_x1, m_y1;
};

/**
 * @brief
 * \f$ f(x) = f(x,y) = f(x,y,z) = ax+b \f$
 */
struct LinearX
{
    /**
     * @brief Construct with two coefficients
     *
     * @param a linear coefficient
     * @param b constant coefficient
     */
    LinearX( double a, double b):a_(a), b_(b){}
    DG_DEVICE
    double operator()(double x)const{ return a_*x+b_;}
    DG_DEVICE
    double operator()( double x, double)const{ return this->operator()(x);}
    DG_DEVICE
    double operator()( double x, double, double)const{ return this->operator()(x);}
   private:
    double a_,b_;
};
/**
 * @brief
 * \f$ f(x,y) = f(x,y,z) = ay+b \f$
 */
struct LinearY
{
    /**
     * @brief Construct with two coefficients
     *
     * @param a linear coefficient
     * @param b constant coefficient
     */
    LinearY( double a, double b):a_(a), b_(b){}
    DG_DEVICE
    double operator()( double, double y, double)const { return a_*y+b_;}
    DG_DEVICE
    double operator()( double, double y)const{ return a_*y+b_;}
  private:
    double a_,b_;
};
/**
 * @brief
 * \f$ f(x,y,z) = az+b \f$
 */
struct LinearZ
{
    /**
     * @brief Construct with two coefficients
     *
     * @param a linear coefficient
     * @param b constant coefficient
     */
    LinearZ( double a, double b):a_(a), b_(b){}
    DG_DEVICE
    double operator()( double, double, double z)const{ return a_*z+b_;}
  private:
    double a_,b_;
};
/**
 * @brief
 * \f$
   f(x,y) = Ae^{-\left(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2}\right)}
   \f$
 */
struct Gaussian
{
    /**
     * @brief Functor returning a %Gaussian
     *
     * @param x0 x-center-coordinate
     * @param y0 y-center-coordinate
     * @param sigma_x x - variance (must be !=0)
     * @param sigma_y y - variance (must be !=0)
     * @param amp Amplitude
     */
    Gaussian( double x0, double y0, double sigma_x, double sigma_y, double amp)
        : m_x0(x0), m_y0(y0), m_sigma_x(sigma_x), m_sigma_y(sigma_y), m_amp(amp){
            assert( m_sigma_x != 0  &&  "sigma_x must not be 0 in Gaussian");
            assert( m_sigma_y != 0  &&  "sigma_y must not be 0 in Gaussian");
    }
    /**
     * @brief Return the value of the %Gaussian
     *
     * \f[
       f(x,y) = Ae^{-\left(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2}\right)}
       \f]
     * @param x x - coordinate
     * @param y y - coordinate
     *
     * @return gaussian
     */
    DG_DEVICE
    double operator()(double x, double y) const
    {
        return  m_amp*
                   exp( -((x-m_x0)*(x-m_x0)/2./m_sigma_x/m_sigma_x +
                          (y-m_y0)*(y-m_y0)/2./m_sigma_y/m_sigma_y) );
    }
    /**
     * @brief Return the value of the %Gaussian
     * \f[
       f(x,y,z) = Ae^{-\left(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2}\right)}
       \f]
     * @param x x - coordinate
     * @param y y - coordinate
     *
     * @return gaussian
     */
    DG_DEVICE
    double operator()(double x, double y, double) const
    {
        return  this->operator()(x,y);
    }
  private:
    double  m_x0, m_y0, m_sigma_x, m_sigma_y, m_amp;

};

/**
 * @brief
 * \f$
   f(x,y) = \begin{cases}
   Ae^{1 + \left(\frac{(x-x_0)^2}{\sigma_x^2} + \frac{(y-y_0)^2}{\sigma_y^2} - 1\right)^{-1}} \text{ if } \frac{(x-x_0)^2}{\sigma_x^2} + \frac{(y-y_0)^2}{\sigma_y^2} < 1\\
   0 \text{ else}
   \end{cases}
   \f$

   A bump that drops to zero and is infinitely continuously differentiable
 */
struct Cauchy
{
    /**
     * @brief A blob that drops to zero
     *
     * @param x0 x-center-coordinate
     * @param y0 y-center-coordinate
     * @param sigma_x radius in x (must be !=0)
     * @param sigma_y radius in y (must be !=0)
     * @param amp Amplitude
     */
    Cauchy( double x0, double y0, double sigma_x, double sigma_y, double amp): x0_(x0), y0_(y0), sigmaX_(sigma_x), sigmaY_(sigma_y), amp_(amp){
        assert( sigma_x != 0  &&  "sigma_x must be !=0 in Cauchy");
        assert( sigma_y != 0  &&  "sigma_y must be !=0 in Cauchy");
    }
    DG_DEVICE
    double operator()(double x, double y )const{
        double xbar = (x-x0_)/sigmaX_;
        double ybar = (y-y0_)/sigmaY_;
        if( xbar*xbar + ybar*ybar < 1.)
            return amp_*exp( 1. +  1./( xbar*xbar + ybar*ybar -1.) );
        return 0.;
    }
    bool inside( double x, double y)const
    {
        double xbar = (x-x0_)/sigmaX_;
        double ybar = (y-y0_)/sigmaY_;
        if( xbar*xbar + ybar*ybar < 1.)
            return true;
        return false;
    }

    double dx( double x, double y )const{
        double xbar = (x-x0_)/sigmaX_;
        double ybar = (y-y0_)/sigmaY_;
        double temp = sigmaX_*(xbar*xbar + ybar*ybar  - 1.);
        return -2.*(x-x0_)*this->operator()(x,y)/temp/temp;
    }
    double dxx( double x, double y)const{
        double temp = sigmaY_*sigmaY_*(x-x0_)*(x-x0_) + sigmaX_*sigmaX_*((y-y0_)*(y-y0_) - sigmaY_*sigmaY_);
        double bracket = sigmaX_*sigmaX_*((y-y0_)*(y-y0_)-sigmaY_*sigmaY_)*sigmaX_*sigmaX_*((y-y0_)*(y-y0_)-sigmaY_*sigmaY_)
            -3.*sigmaY_*sigmaY_*sigmaY_*sigmaY_*(x-x0_)*(x-x0_)*(x-x0_)*(x-x0_)
            -2.*sigmaY_*sigmaY_*sigmaX_*sigmaX_*(x-x0_)*(x-x0_)*(y-y0_)*(y-y0_);
        return -2.*sigmaX_*sigmaX_*sigmaY_*sigmaY_*sigmaY_*sigmaY_*this->operator()(x,y)*bracket/temp/temp/temp/temp;
    }
    double dy( double x, double y)const{
        double xbar = (x-x0_)/sigmaX_;
        double ybar = (y-y0_)/sigmaY_;
        double temp = sigmaY_*(xbar*xbar + ybar*ybar  - 1.);
        return -2.*(y-y0_)*this->operator()(x,y)/temp/temp;
    }
    double dyy( double x, double y)const{
        double temp = sigmaX_*sigmaX_*(y-y0_)*(y-y0_) + sigmaY_*sigmaY_*((x-x0_)*(x-x0_) - sigmaX_*sigmaX_);
        double bracket = sigmaY_*sigmaY_*((x-x0_)*(x-x0_)-sigmaX_*sigmaX_)*sigmaY_*sigmaY_*((x-x0_)*(x-x0_)-sigmaX_*sigmaX_)
            -3.*sigmaX_*sigmaX_*sigmaX_*sigmaX_*(y-y0_)*(y-y0_)*(y-y0_)*(y-y0_)
            -2.*sigmaX_*sigmaX_*sigmaY_*sigmaY_*(y-y0_)*(y-y0_)*(x-x0_)*(x-x0_);
        return -2.*sigmaY_*sigmaY_*sigmaX_*sigmaX_*sigmaX_*sigmaX_*this->operator()(x,y)*bracket/temp/temp/temp/temp;
    }
    double dxy( double x, double y )const{
        double xbar = (x-x0_)/sigmaX_;
        double ybar = (y-y0_)/sigmaY_;
        double temp = (xbar*xbar + ybar*ybar  - 1.);
        return 8.*xbar*ybar*this->operator()(x,y)/temp/temp/temp/sigmaX_/sigmaY_
            + 4.*xbar*ybar*this->operator()(x,y)/temp/temp/temp/temp/sigmaX_/sigmaY_
;
    }
    private:
    double x0_, y0_, sigmaX_, sigmaY_, amp_;
};

/**
 * @brief
 * \f$
   f(x,y) = \begin{cases}
   Ae^{1 + \left(\frac{(x-x_0)^2}{\sigma_x^2} - 1\right)^{-1}} \text{ if } \frac{(x-x_0)^2}{\sigma_x^2} < 1\\
   0 \text{ else}
   \end{cases}
   \f$

   A bump that drops to zero and is infinitely continuously differentiable
 */
struct CauchyX
{
    /**
     * @brief A 1D-blob that drops to zero
     *
     * @param x0 x-center-coordinate
     * @param sigma_x radius in x (must be !=0)
     * @param amp Amplitude
     */
    CauchyX( double x0,  double sigma_x, double amp): x0_(x0), sigmaX_(sigma_x),  amp_(amp){
        assert( sigma_x != 0  &&  "sigma_x must be !=0 in Cauchy");
    }
    DG_DEVICE
    double operator()(double x, double )const{
        double xbar = (x-x0_)/sigmaX_;
        if( xbar*xbar  < 1.)
            return amp_*exp( 1. +  1./( xbar*xbar -1.) );
        return 0.;
    }
    bool inside( double x, double)const
    {
        double xbar = (x-x0_)/sigmaX_;
        if( xbar*xbar < 1.)
            return true;
        return false;
    }
    private:
    double x0_, sigmaX_,  amp_;
};

/**
* @brief
* \f$
f(x,y,z) = Ae^{-\left(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2} + \frac{(z-z_0)^2}{2\sigma_z^2}\right)}
\f$
*/
struct Gaussian3d
{
    /**
     * @brief Functor returning a %Gaussian
     *
     * @param x0 x-center-coordinate
     * @param y0 y-center-coordinate
     * @param z0 z-center-coordinate
     * @param sigma_x x - variance (must be !=0)
     * @param sigma_y y - variance (must be !=0)
     * @param sigma_z z - variance (must be !=0)
     * @param amp Amplitude
     */
    Gaussian3d( double x0, double y0, double z0, double sigma_x, double sigma_y, double sigma_z, double amp)
        : m_x0(x0), m_y0(y0), m_z0(z0), m_sigma_x(sigma_x), m_sigma_y(sigma_y), m_sigma_z(sigma_z), m_amp(amp){
            assert( m_sigma_x != 0  &&  "sigma_x must be !=0 in Gaussian3d");
            assert( m_sigma_y != 0  &&  "sigma_y must be !=0 in Gaussian3d");
            assert( m_sigma_z != 0  &&  "sigma_z must be !=0 in Gaussian3d");
    }
    /**
     * @brief Return a 2d %Gaussian
     *
     * \f[
       f(x,y) = Ae^{-(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2})}
       \f]
     * @param x x - coordinate
     * @param y y - coordinate
     *
     * @return gaussian
     */
    DG_DEVICE
    double operator()(double x, double y) const
    {
        return  m_amp*
                   exp( -((x-m_x0)*(x-m_x0)/2./m_sigma_x/m_sigma_x +
                          (y-m_y0)*(y-m_y0)/2./m_sigma_y/m_sigma_y) );
    }
    /**
     * @brief Return the value of the %Gaussian
     *
     * \f[
       f(x,y) = Ae^{-(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2}+\frac{(z-z_0)^2}{2\sigma_z^2})}
       \f]
     * @param x x - coordinate
     * @param y y - coordinate
     * @param z z - coordinate
     *
     * @return gaussian
     */
    DG_DEVICE
    double operator()(double x, double y, double z) const
    {
        return  m_amp*
                exp( -((x-m_x0)*(x-m_x0)/2./m_sigma_x/m_sigma_x +
                       (z-m_z0)*(z-m_z0)/2./m_sigma_z/m_sigma_z +
                       (y-m_y0)*(y-m_y0)/2./m_sigma_y/m_sigma_y) );
    }
  private:
    double  m_x0, m_y0, m_z0, m_sigma_x, m_sigma_y, m_sigma_z, m_amp;

};
/**
 * @brief
 * \f$
   f(x,y) = Ae^{-\frac{(x-x_0)^2}{2\sigma_x^2} }
   \f$
 */
struct GaussianX
{
    /**
     * @brief A %Gaussian in x
     *
     * @param x0 x-center-coordinate
     * @param sigma_x x - variance (must be !=0)
     * @param amp Amplitude
     */
    GaussianX( double x0, double sigma_x, double amp)
        :m_x0(x0), m_sigma_x(sigma_x), m_amp(amp){
            assert( m_sigma_x != 0  &&  "sigma_x must be !=0 in GaussianX");
    }
    DG_DEVICE
    double operator()(double x) const
    {
        return  m_amp* exp( -((x-m_x0)*(x-m_x0)/2./m_sigma_x/m_sigma_x ));
    }
    DG_DEVICE
    double operator()(double x, double) const
    {
        return this->operator()(x);
    }
    DG_DEVICE
    double operator()(double x, double, double) const
    {
        return this->operator()(x);
    }
  private:
    double  m_x0, m_sigma_x, m_amp;

};
/**
 * @brief
 * \f$
   f(x,y) = Ae^{-\frac{(y-y_0)^2}{2\sigma_y^2}}
   \f$
 */
struct GaussianY
{
    /**
     * @brief Functor returning a gaussian
     *
     * @param y0 y-center-coordinate
     * @param sigma_y y - variance (must be !=0)
     * @param amp Amplitude
     */
    GaussianY( double y0, double sigma_y, double amp)
        : m_y0(y0), m_sigma_y(sigma_y), m_amp(amp){
            assert( m_sigma_y != 0  &&  "sigma_x must be !=0 in GaussianY");
    }
    /**
     * @brief Return the value of the gaussian
     *
     * \f[
       f(x,y) = Ae^{-\frac{(y-y_0)^2}{2\sigma_y^2}}
       \f]
     * @param y y - coordinate
     *
     * @return gaussian
     */
    DG_DEVICE
    double operator()(double, double y) const
    {
        return  m_amp*exp( -((y-m_y0)*(y-m_y0)/2./m_sigma_y/m_sigma_y) );
    }
  private:
    double  m_y0, m_sigma_y, m_amp;

};
/**
 * @brief
 * \f$
   f(x,y,z) = Ae^{-\frac{(z-z_0)^2}{2\sigma_z^2}}
   \f$
 */
struct GaussianZ
{
    /**
     * @brief Functor returning a gaussian
     *
     * @param z0 z-center-coordinate
     * @param sigma_z z - variance (must be !=0)
     * @param amp Amplitude
     */
    GaussianZ( double z0, double sigma_z, double amp)
        : m_z0(z0), m_sigma_z(sigma_z), m_amp(amp){
            assert( m_sigma_z != 0  &&  "sigma_z must be !=0 in GaussianZ");
    }
    /**
     * @brief Return the value of the gaussian
     *
     * \f[
       f(z) = Ae^{-\frac{(z-z_0)^2}{2\sigma_z^2}}
       \f]
     * @param z z - coordinate
     *
     * @return gaussian
     */
    DG_DEVICE
    double operator()( double z) const
    {
        return  m_amp*exp( -((z-m_z0)*(z-m_z0)/2./m_sigma_z/m_sigma_z) );
    }
    /**
     * @brief Return the value of the gaussian
     *
     * \f[
       f(x,y,z) = Ae^{-\frac{(z-z_0)^2}{2\sigma_z^2}}
       \f]
     * @param z z - coordinate
     *
     * @return gaussian
     */
    DG_DEVICE
    double operator()(double, double, double z) const
    {
        return  m_amp*exp( -((z-m_z0)*(z-m_z0)/2./m_sigma_z/m_sigma_z) );
    }
  private:
    double  m_z0, m_sigma_z, m_amp;

};
/**
 * @brief
 * \f$ f(x,y) = \lambda \ln{(\cosh{(x/\lambda) } +\epsilon \cos(y/\lambda)) } \f$
 */
struct IslandXY
{
    /**
     * @brief Construct Island
     *
     * @param lambda amplitude (must be != 0)
     * @param eps y-amplitude
     */
     IslandXY( double lambda, double eps):lambda_(lambda), eps_(eps){
         assert( lambda != 0 && "Lambda parameter in IslandXY must not be zero!");
     }
    /**
     * @brief Return profile
     *
     * @param x x - coordinate
     * @param y y - coordinate
     * @return \f$ f(x,y)\f$
     */
    DG_DEVICE
    double operator()( double x, double y)const{ return lambda_*log(cosh(x/lambda_)+eps_*cos(y/lambda_));}
  private:
    double lambda_,eps_;
};
/**
 * @brief \f$ f(x,y) =B+ A \sin(k_x x) \sin(k_y y) \f$
 */
struct SinXSinY
{
    /**
     * @brief Construct
     *
     * @param amp amplitude A
     * @param bamp backgroundamp B
     * @param kx  kx
     * @param ky  ky
     */
    SinXSinY( double amp, double bamp, double kx, double ky):amp_(amp), bamp_(bamp),kx_(kx),ky_(ky){}
    /**
     * @brief Return profile
     *
     * @param x x - coordinate
     * @param y y - coordinate

     * @return \f$ f(x,y)\f$
     */
    DG_DEVICE
    double operator()( double x, double y)const{ return bamp_+amp_*sin(x*kx_)*sin(y*ky_);}
  private:
    double amp_,bamp_,kx_,ky_;
};
/**
 * @brief
 * \f$ f(x,y) =B+ A \cos(k_x x) \cos(k_y y) \f$
 */
struct CosXCosY
{
    /**
     * @brief Construct
     *
     * @param amp amplitude A
     * @param bamp backgroundamp B
     * @param kx  kx
     * @param ky  ky
     */
    CosXCosY( double amp, double bamp, double kx, double ky):amp_(amp), bamp_(bamp),kx_(kx),ky_(ky){}
    /**
     * @brief Return profile
     *
     * @param x x - coordinate
     * @param y y - coordinate

     * @return \f$ f(x,y)\f$
     */
    DG_DEVICE
    double operator()( double x, double y)const{ return bamp_+amp_*cos(x*kx_)*cos(y*ky_);}
  private:
    double amp_,bamp_,kx_,ky_;
};
/**
 * @brief
 * \f$ f(x,y) =B+ A \sin(k_x x) \cos(k_y y) \f$
 */
struct SinXCosY
{
    /**
     * @brief Construct
     *
     * @param amp amplitude
     * @param bamp backgroundamp
     * @param kx  kx
     * @param ky  ky
     */
    SinXCosY( double amp, double bamp, double kx, double ky):amp_(amp), bamp_(bamp),kx_(kx),ky_(ky){}
    /**
     * @brief Return profile
     *
     * @param x x - coordinate
     * @param y y - coordinate

     * @return \f$ f(x,y)\f$
     */
    DG_DEVICE
    double operator()( double x, double y)const{ return bamp_+amp_*sin(x*kx_)*cos(y*ky_);}
  private:
    double amp_,bamp_,kx_,ky_;
};
/**
 * @brief
 * \f$ f(x) = f(x,y) = f(x,y,z) =B+ A \sin(k_x x) \f$
 */
struct SinX
{
    /**
     * @brief Construct
     *
     * @param amp amplitude A
     * @param bamp backgroundamp B
     * @param kx  kx
     */
    SinX( double amp, double bamp, double kx):amp_(amp), bamp_(bamp),kx_(kx){}
    DG_DEVICE
    double operator()( double x)const{ return bamp_+amp_*sin(x*kx_);}
    DG_DEVICE
    double operator()( double x, double)const{ return this->operator()(x);}
    DG_DEVICE
    double operator()( double x, double, double)const{ return this->operator()(x);}
  private:
    double amp_,bamp_,kx_;
};
/**
 * @brief
 * \f$ f(x,y) =B+ A \sin(k_y y) \f$
 */
struct SinY
{
    /**
     * @brief Construct
     *
     * @param amp amplitude
     * @param bamp backgroundamp
     * @param ky  ky
     */
    SinY( double amp, double bamp, double ky):amp_(amp), bamp_(bamp),ky_(ky){}
    DG_DEVICE
    double operator()( double, double y)const{ return bamp_+amp_*sin(y*ky_);}
  private:
    double amp_,bamp_,ky_;
};
/**
 * @brief
 * \f$ f(x,y) =B+ A \cos(k_y y) \f$
 */
struct CosY
{
    /**
     * @brief Construct
     *
     * @param amp amplitude A
     * @param bamp backgroundamp B
     * @param ky  ky
     */
    CosY( double amp, double bamp, double ky):amp_(amp), bamp_(bamp),ky_(ky){}
    DG_DEVICE
    double operator()( double, double y)const{ return bamp_+amp_*cos(y*ky_);}
  private:
    double amp_,bamp_,ky_;
};
/**
 * @brief
 * \f$ f(x,y) =A/\cosh^2(k_x x) \f$
 */
struct InvCoshXsq
{
    /**
     * @brief Construct with two coefficients
     *
     * @param amp amplitude
     * @param kx  kx
     */
    InvCoshXsq( double amp, double kx):m_amp(amp), m_kx(kx){}
    DG_DEVICE
    double operator()( double x)const{ return m_amp/cosh(x*m_kx)/cosh(x*m_kx);}
    DG_DEVICE
    double operator()( double x, double)const{ return this->operator()(x);}
    DG_DEVICE
    double operator()( double x, double, double)const{ return this->operator()(x);}
  private:
    double m_amp, m_kx;
};
/**
 * @brief
 * \f$ f(x) = f(x,y) = f(x,y,z) = B + A(1 - \sin(k_xx )) \f$
 */
struct SinProfX
{
    /**
     * @brief Construct
     *
     * @param amp amplitude A
     * @param bamp backgroundamp B
     * @param kx  kx
     */
    SinProfX( double amp, double bamp, double kx):m_amp(amp), m_bamp(bamp),m_kx(kx){}
    DG_DEVICE
    double operator()( double x)const{ return m_bamp+m_amp*(1.-sin(x*m_kx));}
    DG_DEVICE
    double operator()( double x, double)const{ return this->operator()(x);}
    DG_DEVICE
    double operator()( double x, double, double)const{ return this->operator()(x);}
  private:
    double m_amp, m_bamp, m_kx;
};
/**
 * @brief
 * \f$ f(x) = f(x,y) = f(x,y,z) = A\exp(-x/L_n) + B \f$
 */
struct ExpProfX
{
    /**
     * @brief Construct with three coefficients
     *
     * @param amp amplitude A
     * @param bamp background amplitude B (choose zero for constant gradient length
     * @param ln  gradient lenght L_n (must be !=0)
     */
    ExpProfX( double amp, double bamp, double ln):m_amp(amp),m_bamp(bamp),m_ln(ln){
        assert( ln!=0 && "ln parameter must be != 0 in ExpProfX!");
    }
    DG_DEVICE
    double operator()( double x)const{ return m_bamp+m_amp*exp(-x/m_ln);}
    DG_DEVICE
    double operator()( double x, double)const{ return this->operator()(x);}
    DG_DEVICE
    double operator()( double x, double, double)const{ return this->operator()(x);}
  private:
    double m_amp, m_bamp, m_ln;
};


/**
 * @brief
     \f$ f(\psi) = \begin{cases}
 1 \text{ if } \psi < \psi_{\max}\\
 0 \text{ if } \psi > (\psi_{\max} + 4\alpha) \\
 \exp\left( - \frac{(\psi - \psi_{\max})^2}{2\alpha^2}\right), \text{ else}
 \end{cases}
   \f$

   One up to \c psimax, then a %Gaussian down to zero
 */
struct GaussianDamping
{
    GaussianDamping( double psimax, double alpha):
        m_psimax(psimax), m_alpha(alpha) {
            assert( alpha!= 0 && "Damping width in GaussianDamping must not be zero");
        }
    DG_DEVICE
    double operator()(double psi)const
    {
        if( psi > m_psimax + 4.*m_alpha) return 0.;
        if( psi < m_psimax) return 1.;
        return exp( -( psi-m_psimax)*( psi-m_psimax)/2./m_alpha/m_alpha);
    }
    private:
    double m_psimax, m_alpha;
};
/**
 * @brief
 * \f$ f(x) = B + 0.5 A(1+ \text{sign} \tanh((x-x_b)/\alpha ) ) \f$
 *
 * An approximation to Heaviside using tanh
 */
struct TanhProfX {
    /**
     * @brief Construct with xb, width and sign
     *
     * @param xb boundary value
     * @param width damping width \c alpha (must be !=0)
     * @param sign sign of the Tanh, defines the damping direction
     * @param bgamp background amplitude \c B
     * @param profamp profile amplitude \c A
     * @note When sign is positive the function leaves the positive and damps the negative and vice versa when sign is negative the function leaves the negative and damps the positive.
     */
    TanhProfX(double xb, double width, int sign =1,double bgamp = 0.,
        double profamp = 1.) :
        xb_(xb),w_(width), s_(sign),bga_(bgamp),profa_(profamp)  {
            assert( width != 0&& "Width in TanhProfX must not be zero!");
    }
    DG_DEVICE
    double operator() (double x)const
    {
        return profa_*0.5*(1.+s_*tanh((x-xb_)/w_))+bga_;
    }
    DG_DEVICE
    double operator()( double x, double)const{ return this->operator()(x);}
    DG_DEVICE
    double operator()( double x, double, double)const{ return this->operator()(x);}
    private:
    double xb_;
    double w_;
    int s_;
    double bga_;
    double profa_;
};

/**
 * @brief \f$ f(x) = \begin{cases}
     0 \text{ if } x < x_b-a \\
        ((16 a^3 - 29 a^2 (x - x_b) + 20 a (x - x_b)^2 - 5 (x - x_b)^3) (a + x -
   x_b)^4)/(32 a^7) \text{ if } |x-x_b| < a \\
        1  \text{ if } x > x_b + a
     \end{cases}\f$

 An approximation to Heaviside using polynomials.
     This function is 3 times continuously differentiable, takes the value 0.5 at xb and
     has a transition width a on both sides of xb.
 */
struct PolynomialHeaviside {
    /**
     * @brief Construct with xb, width and sign
     *
     * @param xb boundary value
     * @param a transition width (must be != 0)
     * @param sign either +1 (original Heaviside) or -1 (the function is mirrored at the \c x=xb axis: f(2xb-x))
     * @note When sign is positive the function leaves the positive and damps the negative and vice versa when sign is negative the function leaves the negative and damps the positive.
     */
    PolynomialHeaviside(double xb, double a, int sign = +1) :
        x0(xb), a(a), m_s(sign){
            assert( a!=0 && "PolynomialHeaviside width must not be zero");
    }
    DG_DEVICE
    double operator() (double x)const
    {
        if( m_s == -1) x = 2*x0-x; //mirror
        if ( x < x0-a) return 0;
        if ( x > x0+a) return 1;
        return ((16.*a*a*a - 29.*a*a*(x - x0)
               + 20.*a*(x - x0)*(x - x0)
               - 5.*(x - x0)*(x-x0)*(x-x0))
               *(a + x - x0)*(a + x - x0)
               *(a + x - x0)*(a + x - x0))/(32.*a*a*a * a*a*a*a);
    }
    DG_DEVICE
    double operator()( double x, double)const{ return this->operator()(x);}
    DG_DEVICE
    double operator()( double x, double, double)const{ return this->operator()(x);}
    private:
    double x0, a;
    int m_s;
};

/**
 * @brief
     \f$ f(x) = \begin{cases}
     0 \text{ if } x < x_l-a_l \\
        ((16 a_l^3 - 29 a_l^2 (x - x_l) + 20 a_l (x - x_l)^2 - 5 (x - x_l)^3) (a_l + x -
   x_l)^4)/(32 a_l^7) \text{ if } |x-x_l| < a_l \\
        1  \text{ if } x_l + a_l < x < x_r-a_r \\
        ((16 a_r^3 - 29 a_r^2 (x - x_r) + 20 a_r (x - x_r)^2 - 5 (x - x_r)^3) (a_r + x -
   x_l)^4)/(32 a_r^7) \text{ if } |x-x_r| < a_r \\
   0 \text{ if } x > x_r + a_r
     \end{cases}\f$

 An approximation to the Rectangle function using polynomials
     Basically just the product of two PolynomialHeaviside functions

     This function is 3 times continuously differentiable, takes the value 0.5 at xl and xr and
     has a transition width a_l on both sides of xl and a width a_r on both sides of xr.
 */
struct PolynomialRectangle {
    /**
     * @brief Construct with xb, width and sign
     *
     * @param xl left boundary value
     * @param al left transition width (must be != 0)
     * @param xr right boundary value
     * @param ar right transition width (must be != 0)
     */
    PolynomialRectangle(double xl, double al, double xr, double ar) :
        m_hl( xl, al, +1), m_hr( xr, ar, -1) {
        assert( xl < xr && "left boundary must be left of right boundary");
    }
    DG_DEVICE
    double operator() (double x)const
    {
        return m_hl(x)*m_hr(x);
    }
    DG_DEVICE
    double operator()( double x, double)const{ return this->operator()(x);}
    DG_DEVICE
    double operator()( double x, double, double)const{ return this->operator()(x);}
    private:
    PolynomialHeaviside m_hl, m_hr;
};

/**
 * @brief \f$ f(x) = \begin{cases}
     x_b \text{ if } x < x_b-a \\
     x_b + ((35 a^3 - 47 a^2 (x - x_b) + 25 a (x - x_b)^2 - 5 (x - x_b)^3) (a + x - x_b)^5)/(256 a^7)
        \text{ if } |x-x_b| < a \\
        x  \text{ if } x > x_b + a
     \end{cases}\f$
 The integral of PolynomialHeaviside approximates xH(x)

     This function is 4 times continuously differentiable,
     has a transition width \c a on both sides of \c xb, where it transitions from the
     constant \c xb to the linear function \c x.
 */
struct IPolynomialHeaviside {
    /**
     * @brief Construct with xb, width and sign
     *
     * @param xb boundary value
     * @param a transition width (must be != 0)
     * @param sign either +1 (original) or -1 (the function is point mirrored at \c x=xb: 2*xb-f(2xb-x))
     */
    IPolynomialHeaviside(double xb, double a, int sign = +1) :
        x0(xb), a(a), m_s(sign){
            assert( a!=0 && "IPolynomialHeaviside width must not be zero");
        }
    DG_DEVICE
    double operator() (double x)const
    {
        if( m_s == -1) x = 2*x0-x; //mirror
        double result;
        if ( x < x0-a) result =  x0;
        else if ( x > x0+a) result =  x;
        else
            result =  x0 + ((35.* a*a*a - 47.* a*a*(x - x0) + 25.*a*(x - x0)*(x-x0)
                - 5.*(x - x0)*(x-x0)*(x-x0))
                *(a+x-x0)*(a+x-x0)*(a+x-x0)*(a+x-x0)*(a+x-x0))
            /(256.*a*a*a * a*a*a*a);
        if ( m_s == +1) return result;
        return 2*x0 - result;

    }
    DG_DEVICE
    double operator()( double x, double)const{ return this->operator()(x);}
    DG_DEVICE
    double operator()( double x, double, double)const{ return this->operator()(x);}
    private:
    double x0, a;
    int m_s;
};

/**
 * @brief \f$ f(x) = \begin{cases}
     0 \text{ if } x < x_b-a || x > x_b+a \\
     (35 (a + x - x_b)^3 (a - x + x_b)^3)/(32 a^7)
        \text{ if } |x-x_b| < a
     \end{cases}\f$
     The derivative of PolynomialHeaviside approximates delta(x)

     This function is 2 times continuously differentiable, is symmetric around \c xb
     and has a width \c a on both sides of \c x0.
     The integral over this function yields 1.
 */
struct DPolynomialHeaviside {
    /**
     * @brief Construct with xb, width and sign
     *
     * @param xb boundary value
     * @param a transition width ( must be !=0)
     *
     * [unnamed-parameter] either +1 (original) or -1 (the function is mirrored at \c x=x0)
     * (since this function is symmetric this parameter is ignored, it's there to be
     * consistent with PolynomialHeaviside)
     */
    DPolynomialHeaviside(double xb, double a, int = +1) :
        x0(xb), a(a){
            assert( a!=0 && "DPolynomialHeaviside width must not be zero");
    }
    DG_DEVICE
    double operator() (double x)const
    {
        if ( (x < x0-a) || (x > x0+a)) return 0;
        return (35.*(a+x-x0)*(a+x-x0)*(a+x-x0)*(a-x+x0)*(a-x+x0)*(a-x+x0))
            /(32.*a*a*a * a*a*a*a);
    }
    DG_DEVICE
    double operator()( double x, double)const{ return this->operator()(x);}
    DG_DEVICE
    double operator()( double x, double, double)const{ return this->operator()(x);}
    private:
    double x0, a;
};

/**
 * @brief \f$ f(i) = \begin{cases}
    1 \text{ if } \eta < \eta_c \\
    \exp\left( -\alpha  \left(\frac{\eta-\eta_c}{1-\eta_c} \right)^{2s}\right) \text { if } \eta \geq \eta_c \\
    0 \text{ else} \\
    \eta=\frac{i}{1-n}
    \end{cases}\f$

    where n is the number of polynomial coefficients

    This function is s times continuously differentiable everywhere
    @sa Its main use comes from the application in dg::ModalFilter
 */
struct ExponentialFilter
{
    /**
     * @brief Create exponential filter \f$ \begin{cases}
    1 \text{ if } \eta < \eta_c \\
    \exp\left( -\alpha  \left(\frac{\eta-\eta_c}{1-\eta_c} \right)^{2s}\right) \text { if } \eta \geq \eta_c \\
    0 \text{ else} \\
    \eta := \frac{i}{n-1}
    \end{cases}\f$
     *
     * @param alpha damping for the highest mode is \c exp( -alpha)
     * @param eta_c cutoff frequency (0<eta_c<1), 0.5 or 0 are good starting values
     * @param order 8 or 16 are good values
     * @param n The number of polynomial coefficients
     */
    ExponentialFilter( double alpha, double eta_c, unsigned order, unsigned n):
        m_alpha(alpha), m_etac(eta_c), m_s(order), m_n(n) {}
    double operator()( unsigned i) const
    {
        double eta = (double)i/(double)(m_n-1);
        if( m_n == 1) eta = 0.;
        if( eta < m_etac)
            return 1.;
        if( eta <= 1.)
            return exp( -m_alpha*pow( (eta-m_etac)/(1.-m_etac), 2*m_s));
        return 0;
    }
    private:
    double m_alpha, m_etac;
    unsigned m_s, m_n;
};

/**
 * @brief
 \f$ f(x,y) = \begin{cases} 2\lambda U J_1(\lambda r) / J_0(\gamma)\cos(\theta) \text{ for } r<R \\
         0 \text{ else}
         \end{cases}
 \f$

 with \f$ r = \sqrt{(x-x_0)^2 + (y-y_0)^2}\f$, \f$
 \theta = \arctan_2( (y-y_), (x-x_0))\f$,
 \f$J_0, J_1\f$ are
 Bessel functions of the first kind of order 0 and 1 and
 \f$\lambda = \gamma/R\f$ with \f$ \gamma = 3.83170597020751231561\f$
 This is the Lamb dipole
 */
struct Lamb
{
    /**
     * @brief Functor returning a Lamb-dipole
     *
     * @param x0 x-center-coordinate
     * @param y0 y-center-coordinate
     * @param R radius of the dipole
     * @param U  speed of the dipole
     */
    Lamb(  double x0, double y0, double R, double U):R_(R), U_(U), x0_(x0), y0_(y0)
    {
        gamma_ = 3.83170597020751231561;
        lambda_ = gamma_/R;
#ifdef _MSC_VER
		j_ = _j0(gamma_);
#else
        j_ = j0( gamma_);
#endif
        //std::cout << r_ <<u_<<x0_<<y0_<<lambda_<<gamma_<<j_<<std::endl;
    }
    /**
     * @brief Return the value of the dipole
     *
     * @param x x - coordinate
     * @param y y - coordinate
     *
     * @return Lamb
     */
    DG_DEVICE
    double operator() (double x, double y)const
    {
        double radius = sqrt( (x-x0_)*(x-x0_) + (y-y0_)*(y-y0_));
        double theta = atan2( (y-y0_),(x-x0_));

        if( radius <= R_)
#ifdef _MSC_VER
			return 2.*lambda_*U_*_j1(lambda_*radius)/j_*cos( theta);
#else
            return 2.*lambda_*U_*j1( lambda_*radius)/j_*cos( theta);
#endif
        return 0;
    }
    /**
     * @brief The total enstrophy of the dipole
     *
     * Analytic formula. True for periodic and dirichlet boundary conditions.
     * @return enstrophy \f$ \pi U^2\gamma^2\f$

     */
    double enstrophy( ) const { return M_PI*U_*U_*gamma_*gamma_;}

    /**
     * @brief The total energy of the dipole
     *
     * Analytic formula. True for periodic and dirichlet boundary conditions.
     * @return  energy \f$ 2\pi R^2U^2\f$
     */
    double energy() const { return 2.*M_PI*R_*R_*U_*U_;}
  private:
    double R_, U_, x0_, y0_, lambda_, gamma_, j_;
};

/**
 * @brief
       \f$f(x,y) =\begin{cases}
       \frac{u_d}{1.2965125} \left(
       r\left(1+\frac{\beta_i^2}{g_i^2}\right)
       - R \frac{\beta_i^2}{g_i^2} \frac{J_1(g_ir/R)}{J_1(g_i)}\right)\cos(\theta) \text{ if } r < R \\
      \frac{u_d}{1.2965125} R \frac{K_1(\beta_i {r}/{R})}{K_1(\beta)} \cos(\theta) \text{ else }
      \end{cases}
      \f$

      Return a 2d vortex function
     * where \f$ i\in \{0,1,2\}\f$ is the mode number and r and \f$\theta\f$ are poloidal coordinates
 with \f$ r = \sqrt{(x-x_0)^2 + (y-y_0)^2}\f$, \f$ \theta = \arctan_2( (y-y_), (x-x_0))\f$,
        \f$ g_0 = 3.831896621 \f$,
        \f$ g_1 = -3.832353624 \f$,
        \f$ g_2 = 7.016\f$,
        \f$ \beta_0 = 0.03827327723\f$,
        \f$ \beta_1 = 0.07071067810 \f$,
        \f$ \beta_2 = 0.07071067810 \f$
        \f$ K_1\f$ is the modified and \f$ J_1\f$ the Bessel function
 */
struct Vortex
{
    /**
     * @brief
     *
     * @param x0 X position
     * @param y0 Y position
     * @param state mode 0,1, or 2
     * @param R characteristic radius of dipole
     * @param u_dipole u_drift/u_dipole = \f$ u_d\f$
     * @param kz multiply by \f$ \cos(k_z z) \f$ in three dimensions
     */
    Vortex( double x0, double y0, unsigned state,
          double R,  double u_dipole, double kz = 0):
        x0_(x0), y0_(y0), s_(state),  R_(R), u_d( u_dipole), kz_(kz){
        g_[0] = 3.831896621;
        g_[1] = -3.832353624;
        g_[2] = 7.016;
        b_[0] = 0.03827327723;
        b_[1] = 0.07071067810 ;
        b_[2] = 0.07071067810 ;
    }
    /**
     * @brief \f$f(x,y) =\begin{cases}
       \frac{u_d}{1.2965125} \left(
       r\left(1+\frac{\beta_i^2}{g_i^2}\right)
       - R \frac{\beta_i^2}{g_i^2} \frac{J_1(g_ir/R)}{J_1(g_i)}\right)\cos(\theta) \text{ if } r < R \\
      \frac{u_d}{1.2965125} R \frac{K_1(\beta_i {r}/{R})}{K_1(\beta)} \cos(\theta) \text{ else }
      \end{cases}
      \f$

      Evaluate the vortex
     * where \f$ i\in \{0,1,2\}\f$ is the mode number and r and \f$\theta\f$ are poloidal coordinates
     * @param x value
     * @param y value
     *
     * @return the above function value
     */
    DG_DEVICE
    double operator()( double x, double y)const
    {
        double r = sqrt( (x-x0_)*(x-x0_)+(y-y0_)*(y-y0_));
        double theta = atan2( y-y0_, x-x0_);
        double beta = b_[s_];
        double norm = 1.2965125;

        if( r/R_<=1.)
            return u_d*(
                      r *( 1 +beta*beta/g_[s_]/g_[s_] )
#ifdef _MSC_VER
                    - R_*  beta*beta/g_[s_]/g_[s_] *_j1(g_[s_]*r/R_)/_j1(g_[s_])
#else
				    - R_ * beta*beta/g_[s_]/g_[s_] * j1(g_[s_]*r/R_)/ j1(g_[s_])
#endif
                    )*cos(theta)/norm;
        return u_d * R_* bessk1(beta*r/R_)/bessk1(beta)*cos(theta)/norm;
        // TODO Can these be replaced by std::cyl_bessel_k? Not sure what to do on device though
    }
    /**
     * @brief \f$f(x,y,z) =\cos(k_z z)\begin{cases}
       \frac{u_d}{1.2965125} \left(
       r\left(1+\frac{\beta_i^2}{g_i^2}\right)
       - R \frac{\beta_i^2}{g_i^2} \frac{J_1(g_ir/R)}{J_1(g_i)}\right)\cos(\theta) \text{ if } r < R \\
      \frac{u_d}{1.2965125} R \frac{K_1(\beta_i {r}/{R})}{K_1(\beta)} \cos(\theta) \text{ else }
      \end{cases}
      \f$

      Evaluate the vortex modulated by a sine wave in z
     * where \f$ i\in \{0,1,2\}\f$ is the mode number and r and \f$\theta\f$ are poloidal coordinates
     * @param x value
     * @param y value
     * @param z value
     *
     * @return the above function value
     */
    DG_DEVICE
    double operator()( double x, double y, double z)const
    {
        return this->operator()(x,y)*cos(kz_*z);
    }
    private:
    // Returns the modified Bessel function K1(x) for positive real x.
    DG_DEVICE
    double bessk1(double x)const
    {
        double y,ans;
        if (x <= 2.0)
        {
            y=x*x/4.0;
            ans = (log(x/2.0)*bessi1(x))+(1.0/x)*(1.0+y*(0.15443144 +
                       y*(-0.67278579+y*(-0.18156897+y*(-0.1919402e-1 +
                       y*(-0.110404e-2+y*(-0.4686e-4)))))));
        }
        else
        {
            y=2.0/x;
            ans = (exp(-x)/sqrt(x))*(1.25331414+y*(0.23498619 +
                      y*(-0.3655620e-1+y*(0.1504268e-1+y*(-0.780353e-2 +
                      y*(0.325614e-2+y*(-0.68245e-3)))))));
        }
        return ans;
    }
    //Returns the modified Bessel function I1(x) for any real x.
    DG_DEVICE
    double bessi1(double x) const
    {
        double ax,ans;
        double y;
        if ((ax=fabs(x)) < 3.75)
        {
            y=x/3.75;
            y*=y;
            ans = ax*(0.5+y*(0.87890594+y*(0.51498869+y*(0.15084934 +
                       y*(0.2658733e-1+y*(0.301532e-2+y*0.32411e-3))))));
        }
        else
        {
            y=3.75/ax;
            ans = 0.2282967e-1+y*(-0.2895312e-1+y*(0.1787654e-1 -
                      y*0.420059e-2)); ans=0.39894228+y*(-0.3988024e-1+
                      y*(-0.362018e-2 +y*(0.163801e-2+y*(-0.1031555e-1+y*ans))));
            ans *= (exp(ax)/sqrt(ax));
        }
        return x < 0.0 ? -ans : ans;
    }
    double x0_, y0_;
    unsigned s_;
    double R_, b_[3], u_d;
    double g_[3];
    double kz_;
};

/**
* @brief \f$f(R,Z) = A B \sum_\vec{k} \sqrt{E_k} \alpha_k \cos{\left(k \kappa_k + \theta_k \right)}
\f$

* A random bath in the R-Z plane
* with \f[ B := \sqrt{\frac{2}{N_{k_R} N_{k_Z}}} \\
        k:=\sqrt{k_R^2 + k_Z^2} \\
        k_R:=2 \pi \left( i -N_{k_R}/2\right)/N_{k_R} \\
        k_Z:=2 \pi \left( j -N_{k_Z}/2\right)/N_{k_Z} \\
        k_0:=2 \pi L_E / N_k\\
        N_k := \sqrt{N_{k_R}^2 + N_{k_Z}^2} \\
        E_k:=\left(4 k k_0/(k+k_0)^2\right)^{\gamma} \\
        \alpha_k := \sqrt{\mathcal{N}_1^2 + \mathcal{N}_2^2} \\
        \theta_k := \arctan{\left(\mathcal{N}_2/\mathcal{N}_1\right)} \\
        \kappa_k(R,Z) := (R-R_{min}) \mathcal{U}_1 + (Z-Z_{min}) \mathcal{U}_2  \\
        \f]
* where \f$\mathcal{N}_{1,2}\f$ are random normal distributed real numbers with a mean of \f$\mu = 0\f$ and a standard deviation of \f$\sigma=1 \f$,  \f$\mathcal{U}_{1,2}\f$ are random uniformly distributed real numbers  \f$\in \left[0, 2 \pi \right) \f$ and \f$ A \f$ is the amplitude.
*/
struct BathRZ{
      /**
     * @brief Functor returning a random field in the RZ-plane or in the first RZ-plane
     *
     * @param N_kR Number of Fourier modes in R direction
     * @param N_kZ Number of Fourier modes in Z direction
     * @param R_min Minimal R (in units of rho_s)
     * @param Z_min Minimal Z (in units of rho_s)
     * @param gamma exponent in the energy function \f$E_k\f$ (typically around 30)
     * @param L_E is the typical eddysize (typically around 5)
     * @param amp Amplitude
     */
    BathRZ( unsigned N_kR, unsigned N_kZ, double R_min, double Z_min, double gamma, double L_E, double amp) :
        N_kR_(N_kR), N_kZ_(N_kZ),
        R_min_(R_min), Z_min_(Z_min),
        gamma_(gamma), L_E_(L_E) , amp_(amp),
        kvec( N_kR_*N_kZ_, 0), sqEkvec(kvec), unif1(kvec), unif2(kvec),
        normal1(kvec), normal2(kvec), alpha(kvec), theta(kvec)
    {
        double N_kR2=(double)(N_kR_*N_kR_);
        double N_kZ2=(double)(N_kZ_*N_kZ_);
        double N_k= sqrt(N_kR2+N_kZ2);

        norm_=sqrt(2./(double)N_kR_/(double)N_kZ_);
        double tpi=2.*M_PI, tpi2=tpi*tpi;
        double k0= tpi*L_E_/N_k;
        double N_kRh = N_kR_/2.;
        double N_kZh = N_kZ_/2.;

        std::minstd_rand generator;
        std::normal_distribution<double> ndistribution( 0.0, 1.0); // ( mean, stddev)
        std::uniform_real_distribution<double> udistribution(0.0,tpi); //between [0 and 2pi)
        for (unsigned j=1;j<=N_kZ_;j++)
        {
            double kZ2=tpi2*(j-N_kZh)*(j-N_kZh)/(N_kZ2);
            for (unsigned i=1;i<=N_kR_;i++)
            {
                double kR2=tpi2*(i-N_kRh)*(i-N_kRh)/(N_kR2);
                int z=(j-1)*(N_kR_)+(i-1);
                kvec[z]= sqrt(kR2 + kZ2);  //radial k number
                sqEkvec[z]=pow(kvec[z]*4.*k0/(kvec[z]+k0)/(kvec[z]+k0),gamma_/2.); //Energie in k space with max at 1.
                unif1[z]=cos(udistribution(generator));
                unif2[z]=sin(udistribution(generator));
                normal1[z]=ndistribution(generator);
                normal2[z]=ndistribution(generator);
                alpha[z]=sqrt(normal1[z]*normal1[z]+normal2[z]*normal2[z]);
                theta[z]=atan2(normal2[z],normal1[z]);
            }
        }

    }
    /**
     * @brief Return the value of the Bath
     *
       \f[f(R,Z) = A B \sum_\vec{k} \sqrt{E_k} \alpha_k \cos{\left(k \kappa_k + \theta_k \right)}
       \f]
     * with \f[ \mathcal{N} := \sqrt{\frac{2}{N_{k_R} N_{k_Z}}} \\
                k:=\sqrt{k_R^2 + k_Z^2} \\
                k_R:=2 \pi \left( i -N_{k_R}/2\right)/N_{k_R} \\
                k_Z:=2 \pi \left( j -N_{k_Z}/2\right)/N_{k_Z} \\
                k_0:=2 \pi L_E / N_k\\
                N_k := \sqrt{N_{k_R}^2 + N_{k_Z}^2} \\
                E_k:=\left(4 k k_0/(k+k_0)^2\right)^{\gamma} \\
                \alpha_k := \sqrt{\mathcal{N}_1^2 + \mathcal{N}_2^2} \\
                \theta_k := \arctan{\left(\mathcal{N}_2/\mathcal{N}_1\right)} \\
                \kappa_k(R,Z) := (R-R_{min}) \mathcal{U}_1 + (Z-Z_{min}) \mathcal{U}_2  \\
                \f]
     * where \f$\mathcal{N}_{1,2}\f$ are random normal distributed real numbers with a mean of \f$\mu = 0\f$ and a standard deviation of \f$\sigma=1 \f$,
     * \f$\mathcal{U}_{1,2}\f$ are random uniformly distributed real numbers  \f$\in \left[0, 2 \pi \right) \f$ and \f$ A \f$ is the amplitude
     * @param R R - coordinate
     * @param Z Z - coordinate
     *
     * @return the above function value
     */
    double operator()(double R, double Z)const
    {
        double f, kappa, RR, ZZ;
        RR=R-R_min_;
        ZZ=Z-Z_min_;
        f=0.;
        for (unsigned j=0;j<N_kZ_;j++)
        {
            for (unsigned i=0;i<N_kR_;i++)
            {
                int z=j*N_kR_+i;
                kappa= RR*unif1[z]+ZZ*unif2[z];
                f+= sqEkvec[z]*alpha[z]*cos(kvec[z]*kappa+theta[z]);
            }
        }
        return amp_*norm_*f;
    }
    /**
     * @brief Return the value of the Bath
     *
       \f[f(R,Z) = A B \sum_\vec{k} \sqrt{E_k} \alpha_k \cos{\left(k \kappa_k + \theta_k \right)}
       \f]
     * with \f[ \mathcal{N} := \sqrt{\frac{2}{N_{k_R} N_{k_Z}}} \\
                k:=\sqrt{k_R^2 + k_Z^2} \\
                k_R:=2 \pi \left( i -N_{k_R}/2\right)/N_{k_R} \\
                k_Z:=2 \pi \left( j -N_{k_Z}/2\right)/N_{k_Z} \\
                k_0:=2 \pi L_E / N_k\\
                N_k := \sqrt{N_{k_R}^2 + N_{k_Z}^2} \\
                E_k:=\left(4 k k_0/(k+k_0)^2\right)^{\gamma} \\
                \alpha_k := \sqrt{\mathcal{N}_1^2 + \mathcal{N}_2^2} \\
                \theta_k := \arctan{\left(\mathcal{N}_2/\mathcal{N}_1\right)} \\
                \kappa_k(R,Z) := (R-R_{min}) \mathcal{U}_1 + (Z-Z_{min}) \mathcal{U}_2  \\
                \f]
     * where \f$\mathcal{N}_{1,2}\f$ are random normal distributed real numbers with a mean of \f$\mu = 0\f$ and a standard deviation of \f$\sigma=1 \f$,
     * \f$\mathcal{U}_{1,2}\f$ are random uniformly distributed real numbers  \f$\in \left[0, 2 \pi \right) \f$ and \f$ A \f$ is the amplitude
     *
     * @param R R - coordinate
     * @param Z Z - coordinate
     *
     * @return the above function value
     */
    double operator()(double R, double Z, double)const {
        double f, kappa;
        double  RR, ZZ;
        RR=R-R_min_;
        ZZ=Z-Z_min_;
        f=0;
        for (unsigned j=0;j<N_kZ_;j++)
        {
            for (unsigned i=0;i<N_kR_;i++)
            {
                int z=(j)*(N_kR_)+(i);
                kappa= RR*unif1[z]+ZZ*unif2[z];
                f+= sqEkvec[z]*alpha[z]*cos(kvec[z]*kappa+theta[z]);
            }
        }
        return amp_*norm_*f;
    }
  private:
    unsigned N_kR_,N_kZ_;
    double R_min_, Z_min_;
    double gamma_, L_E_;
    double amp_;
    double norm_;
    std::vector<double> kvec;
    std::vector<double> sqEkvec;
    std::vector<double> unif1, unif2, normal1,normal2,alpha,theta;
};

/**
 * @brief \f$ f(x,y) = \sum_{i=0}^{M-1} \sum_{j=0}^{N-1} c_{iN+j} x^i y^j  \f$
 *
 * Evaluated using [Horner's method](https://en.wikipedia.org/wiki/Horner%27s_method)
 */
struct Horner2d
{
    ///Initialize 1 coefficient to 1
    Horner2d(): m_c( 1, 1), m_M(1), m_N(1){}

    /**
     * @brief Initialize coefficients and dimensions
     *
     * @param c vector of size MN containing coefficientc c (accessed as c[i*N+j] i.e. y-direction is contiguous)
     * @param M number of polynomials in x
     * @param N number of polynomials in y
     */
    Horner2d( std::vector<double> c, unsigned M, unsigned N): m_c(c), m_M(M), m_N(N){}
    double operator()( double x, double y) const
    {
        std::vector<double> cx( m_M);
        for( unsigned i=0; i<m_M; i++)
            cx[i] = horner( &m_c[i*m_N], m_N, y);
        return horner( &cx[0], m_M, x);
    }
    private:
    double horner( const double * c, unsigned M, double x) const
    {
        double b = c[M-1];
        for( unsigned i=0; i<M-1; i++)
            b = c[M-2-i] + b*x;
        return b;
    }
    std::vector<double> m_c;
    unsigned m_M, m_N;
};


/**
 * @brief Compute a histogram on a 1D grid
 * @tparam container
 */
template <class container = thrust::host_vector<double> >
struct Histogram
{
     /**
     * @brief Construct a histogram from number of bins and an input vector
     * @param g1d grid on which to compute the histogram ( grid.h() is the binwidth)
     * @param in input vector (if grid.x0() < in[i] <grid.x1() it falls in a bin)
     */
    Histogram(const dg::Grid1d& g1d, const std::vector<double>& in) :
    g1d_(g1d),
    in_(in),
    binwidth_(g1d_.h()),
    count_(dg::evaluate(dg::zero,g1d_))
    {
        for (unsigned j=0;j<in_.size();j++)
        {
            unsigned bin =floor( (in_[j]-g1d_.x0())/binwidth_ );
            bin = std::max(bin,(unsigned) 0);
            bin = std::min(bin,(unsigned)(g1d_.size()-1));
            count_[bin ]+=1.;
        }
        //Normalize
        unsigned Ampmax = (unsigned)thrust::reduce( count_.begin(), count_.end(),0.,   thrust::maximum<double>()  );
        dg::blas1::scal(count_,1./Ampmax);

    }

    /**
     * @brief get binwidth
     *
     * @return
     */
    double binwidth() {return binwidth_;}
    /**
     * @brief Access computed histogram
     *
     * @param x
     *
     * @return
     */
    double operator()(double x)const
    {
        unsigned bin = floor((x-g1d_.x0())/binwidth_+0.5);
        bin = std::max(bin,(unsigned) 0);
        bin = std::min(bin,(unsigned)(g1d_.size()-1));
        return count_[bin];
    }

    private:
    dg::Grid1d g1d_;
    const std::vector<double> in_;
    double binwidth_;
    container  count_;
};

/**
 * @brief Compute a histogram on a 2D grid
 * @tparam container
 */
template <class container = thrust::host_vector<double> >
struct Histogram2D
{
     /**
     * @brief Construct a histogram from number of bins and an input vector
     * @param g2d grid on which to compute the histogram ( grid.h() is the binwidth)
     * @param inx input vector in x - direction (if grid.x0() < in[i] <grid.x1() it falls in a bin)
     * @param iny input vector in y - direction (if grid.y0() < in[i] <grid.y1() it falls in a bin)
     */
    Histogram2D(const dg::Grid2d& g2d, const std::vector<double>& inx,const std::vector<double>& iny) :
    g2d_(g2d),
    inx_(inx),
    iny_(iny),
    binwidthx_(g2d_.hx()),
    binwidthy_(g2d_.hy()),
    count_(dg::evaluate(dg::zero,g2d_))
    {

        for (unsigned j=0;j<iny_.size();j++)
        {
            unsigned biny =floor((iny_[j]-g2d_.y0())/binwidthy_) ;
            biny = std::max(biny,(unsigned) 0);
            biny = std::min(biny,(unsigned)(g2d_.Ny()-1));

            unsigned binx =floor((inx_[j]-g2d_.x0())/binwidthx_) ;
            binx = std::max(binx,(unsigned) 0);
            binx = std::min(binx,(unsigned)(g2d_.Nx()-1));
            count_[biny*g2d_.Nx()+binx ]+=1.;

        }
        //Normalize
        unsigned Ampmax =  (unsigned)thrust::reduce( count_.begin(),   count_.end(),0.,thrust::maximum<double>()  );
        dg::blas1::scal(count_,  1./Ampmax);

    }

    /**
     * @brief Access computed histogram
     *
     * @param x
     * @param y
     *
     * @return
     */
    double operator()(double x, double y)const
    {
        unsigned binx = floor((x-g2d_.x0())/binwidthx_+0.5) ;
        binx = std::max(binx,(unsigned) 0);
        binx = std::min(binx,(unsigned)(g2d_.Nx()-1));
        unsigned biny = floor((y-g2d_.y0())/binwidthy_+0.5) ;
        biny = std::max(biny,(unsigned) 0);
        biny = std::min(biny,(unsigned)(g2d_.Ny()-1));
        return count_[biny*g2d_.Nx()+binx ];

    }
    private:
    dg::Grid2d g2d_;
    const std::vector<double> inx_,iny_;
    double binwidthx_,binwidthy_;
    container count_;
};



/**
 * @brief Shortest Distance to a collection of vertical and horizontal lines
 *
 * First determine which line is closest to given point and then determine
 * the exact distance to it
 */
struct WallDistance
{
    /**
     * @brief Allocate lines
     *
     * @param vertical walls R_0, R_1 ...  ( can be arbitrary size)
     * @param horizontal walls Z_0, Z_1 ... ( can be arbitrary size)
     */
    WallDistance( std::vector<double> vertical, std::vector<double> horizontal) :
        m_vertical(vertical), m_horizontal( horizontal) {}
    /**
     * @brief Allocate lines
     *
     * @param walls two vertical (x0, x1) and two horizontal (y0, y1) walls
     */
    WallDistance( dg::Grid2d walls) : m_vertical({walls.x0(), walls.x1()}),
        m_horizontal({walls.y0(), walls.y1()}){}
    /**
     * @brief Distance to closest wall in a box
     */
    double operator() (double R, double Z) const
    {
        std::vector<double> dist( 1, 1e100); //fill in at least one (large) number in case vectors are empty)
        for( auto v : m_vertical)
            dist.push_back(fabs( R-v));
        for( auto h : m_horizontal)
            dist.push_back(fabs( Z-h));
        return *std::min_element( dist.begin(), dist.end());
    }
    private:
    std::vector<double> m_vertical;
    std::vector<double> m_horizontal;
};


///@}
} //namespace dg

