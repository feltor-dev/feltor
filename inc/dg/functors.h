#pragma once

#include <cmath>
//! M_PI is non-standard ... so MSVC complains
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <vector>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/random/normal_distribution.h>
#include "blas1.h"
#include "topology/grid.h"
#include "topology/evaluation.h"
#include "topology/functions.h"
/*!@file
 * Functors to use in dg::evaluate or dg::blas1::transform functions
 */
namespace dg
{

///@addtogroup functions
///@{

/**
 * @brief Absolute maximum
 * \f[ f(x,y) = \max(|x|,|y|)\f]
 *
 */
template <class T = double>
struct AbsMax
{
    /**
     * @brief Return the asbolute maximum
     *
     * @param x left value
     * @param y right value
     *
     * @return absolute maximum
     */
DG_DEVICE
    T operator() ( T x, T y) const
    {
        T absx = x>0 ? x : -x;
        T absy = y>0 ? y : -y;
        return absx > absy ? absx : absy;
    }
};
/**
 * @brief Absolute minimum
 * \f[ f(x,y) = \min(|x|,|y|)\f]
 */
template <class T = double>
struct AbsMin
{
    /**
     * @brief Return the asbolute minimum
     *
     * @param x left value
     * @param y right value
     *
     * @return absolute minimum
     */
DG_DEVICE
    T operator() (T x, T y) const
    {
        T absx = x<0 ? -x : x;
        T absy = y<0 ? -y : y;
        return absx < absy ? absx : -absy;
    }
};

/**
 * @brief Functor returning a gaussian
 * \f[
   f(x,y) = Ae^{-\left(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2}\right)}
   \f]
 */
struct Gaussian
{
    /**
     * @brief Functor returning a gaussian
     *
     * @param x0 x-center-coordinate
     * @param y0 y-center-coordinate
     * @param sigma_x x - variance
     * @param sigma_y y - variance
     * @param amp Amplitude
     * @param kz wavenumber in z direction
     */
    Gaussian( double x0, double y0, double sigma_x, double sigma_y, double amp, double kz = 1.)
        : x00(x0), y00(y0), sigma_x(sigma_x), sigma_y(sigma_y), amplitude(amp), kz_(kz){}
    /**
     * @brief Return the value of the gaussian
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
        return  amplitude*
                   exp( -((x-x00)*(x-x00)/2./sigma_x/sigma_x +
                          (y-y00)*(y-y00)/2./sigma_y/sigma_y) );
    }
    /**
     * @brief Return the value of the gaussian modulated by a cosine
     * \f[
       f(x,y,z) = A\cos(kz)e^{-(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2})}
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
        return  amplitude*cos(kz_*z)*
                   exp( -((x-x00)*(x-x00)/2./sigma_x/sigma_x +
                          (y-y00)*(y-y00)/2./sigma_y/sigma_y) );
    }
  private:
    double  x00, y00, sigma_x, sigma_y, amplitude, kz_;

};

/**
 * @brief A blob that drops to zero
 * \f[
   f(x,y) = Ae^{1 + \left(\frac{(x-x_0)^2}{\sigma_x^2} + \frac{(y-y_0)^2}{\sigma_y^2} - 1\right)^{-1}}
   \f]
 */
struct Cauchy
{
    /**
     * @brief A blob that drops to zero
     *
     * @param x0 x-center-coordinate
     * @param y0 y-center-coordinate
     * @param sigma_x radius in x
     * @param sigma_y radius in y
     * @param amp Amplitude
     */
    Cauchy( double x0, double y0, double sigma_x, double sigma_y, double amp): x0_(x0), y0_(y0), sigmaX_(sigma_x), sigmaY_(sigma_y), amp_(amp){}
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
* @brief The 3d gaussian
* \f[
f(x,y,z) = Ae^{-\left(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2} + \frac{(z-z_0)^2}{2\sigma_z^2}\right)}
\f]
*/
struct Gaussian3d
{
    /**
     * @brief Functor returning a gaussian
     *
     * @param x0 x-center-coordinate
     * @param y0 y-center-coordinate
     * @param z0 z-center-coordinate
     * @param sigma_x x - variance
     * @param sigma_y y - variance
     * @param sigma_z z - variance
     * @param amp Amplitude
     */
    Gaussian3d( double x0, double y0, double z0, double sigma_x, double sigma_y, double sigma_z, double amp)
        : x00(x0), y00(y0), z00(z0), sigma_x(sigma_x), sigma_y(sigma_y), sigma_z(sigma_z), amplitude(amp){}
    /**
     * @brief Return a 2d gaussian
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
        return  amplitude*
                   exp( -((x-x00)*(x-x00)/2./sigma_x/sigma_x +
                          (y-y00)*(y-y00)/2./sigma_y/sigma_y) );
    }
    /**
     * @brief Return the value of the gaussian
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
//         if (z== z00)
//         {
            return  amplitude*
                    exp( -((x-x00)*(x-x00)/2./sigma_x/sigma_x +
                           (z-z00)*(z-z00)/2./sigma_z/sigma_z +
                           (y-y00)*(y-y00)/2./sigma_y/sigma_y) );
//         }
//         else {
//         return 0.;
//         }
    }
  private:
    double  x00, y00, z00, sigma_x, sigma_y, sigma_z, amplitude;

};
/**
 * @brief A Gaussian in x-direction
 * \f[
   f(x,y) = Ae^{-\frac{(x-x_0)^2}{2\sigma_x^2} }
   \f]
 */
struct GaussianX
{
    /**
     * @brief A Gaussian in x
     *
     * @param x0 x-center-coordinate
     * @param sigma_x x - variance
     * @param amp Amplitude
     */
    GaussianX( double x0, double sigma_x, double amp)
        :x00(x0), sigma_x(sigma_x), amplitude(amp){}
    DG_DEVICE
    double operator()(double x) const
    {
        return  amplitude* exp( -((x-x00)*(x-x00)/2./sigma_x/sigma_x ));
    }
    DG_DEVICE
    double operator()(double x, double y) const
    {
        return this->operator()(x);
    }
    DG_DEVICE
    double operator()(double x, double y, double z) const
    {
        return this->operator()(x);
    }
  private:
    double  x00, sigma_x, amplitude;

};
/**
 * @brief A Gaussian in y-direction
 * \f[
   f(x,y) = Ae^{-\frac{(y-y_0)^2}{2\sigma_y^2}}
   \f]
 */
struct GaussianY
{
    /**
     * @brief Functor returning a gaussian
     *
     * @param y0 y-center-coordinate
     * @param sigma_y y - variance
     * @param amp Amplitude
     */
    GaussianY( double y0, double sigma_y, double amp)
        : y00(y0), sigma_y(sigma_y), amplitude(amp){}
    /**
     * @brief Return the value of the gaussian
     *
     * \f[
       f(x,y) = Ae^{-\frac{(y-y_0)^2}{2\sigma_y^2}}
       \f]
     * @param x x - coordinate
     * @param y y - coordinate
     *
     * @return gaussian
     */
    DG_DEVICE
    double operator()(double x, double y) const
    {
        return  amplitude*exp( -((y-y00)*(y-y00)/2./sigma_y/sigma_y) );
    }
  private:
    double  y00, sigma_y, amplitude;

};
/**
 * @brief A Gaussian in z-direction
 * \f[
   f(x,y,z) = Ae^{-\frac{(z-z_0)^2}{2\sigma_z^2}}
   \f]
 */
struct GaussianZ
{
    /**
     * @brief Functor returning a gaussian
     *
     * @param z0 z-center-coordinate
     * @param sigma_z z - variance
     * @param amp Amplitude
     */
    GaussianZ( double z0, double sigma_z, double amp)
        : z00(z0), sigma_z(sigma_z), amplitude(amp){}
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
        return  amplitude*exp( -((z-z00)*(z-z00)/2./sigma_z/sigma_z) );
    }
    /**
     * @brief Return the value of the gaussian
     *
     * \f[
       f(x,y,z) = Ae^{-\frac{(z-z_0)^2}{2\sigma_z^2}}
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
        return  amplitude*exp( -((z-z00)*(z-z00)/2./sigma_z/sigma_z) );
    }
  private:
    double  z00, sigma_z, amplitude;

};
/**
 * @brief Island function
 * \f[ f(x,y) = \lambda \ln{(\cosh{(x/\lambda) } +\epsilon \cos(y/\lambda)) } \f]
 */
struct IslandXY
{
    /**
     * @brief Construct Island
     *
     * @param lambda amplitude
     * @param eps y-amplitude
     */
     IslandXY( double lambda, double eps):lambda_(lambda), eps_(eps){}
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
 * @brief A sin prof in x and y-direction
 * \f[ f(x,y) =B+ A \sin(k_x x) \sin(k_y y) \f]
 */
struct SinXSinY
{
    /**
     * @brief Construct with two coefficients
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
 * @brief A cos prof in x and y-direction
 * \f[ f(x,y) =B+ A \cos(k_x x) \cos(k_y y) \f]
 */
struct CosXCosY
{
    /**
     * @brief Construct with two coefficients
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
 * @brief A sin prof in x- and cos prof in  y-direction
 * \f[ f(x,y) =B+ A \sin(k_x x) \cos(k_y y) \f]
 */
struct SinXCosY
{
    /**
     * @brief Construct with two coefficients
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
 * @brief A sin prof in x-direction
 * \f[ f(x) = f(x,y) = f(x,y,z) =B+ A \sin(k_x x) \f]
 */
struct SinX
{
    /**
     * @brief Construct with two coefficients
     *
     * @param amp amplitude A
     * @param bamp backgroundamp B
     * @param kx  kx
     */
    SinX( double amp, double bamp, double kx):amp_(amp), bamp_(bamp),kx_(kx){}
    DG_DEVICE
    double operator()( double x)const{ return bamp_+amp_*sin(x*kx_);}
    DG_DEVICE
    double operator()( double x, double y)const{ return this->operator()(x);}
    DG_DEVICE
    double operator()( double x, double y, double z)const{ return this->operator()(x);}
  private:
    double amp_,bamp_,kx_;
};
/**
 * @brief A sin prof in y-direction
 * \f[ f(x,y) =B+ A \sin(k_y y) \f]
 */
struct SinY
{
    /**
     * @brief Construct with two coefficients
     *
     * @param amp amplitude
     * @param bamp backgroundamp
     * @param ky  ky
     */
    SinY( double amp, double bamp, double ky):amp_(amp), bamp_(bamp),ky_(ky){}
    DG_DEVICE
    double operator()( double x, double y)const{ return bamp_+amp_*sin(y*ky_);}
  private:
    double amp_,bamp_,ky_;
};
/**
 * @brief A sin prof in x-direction
 * \f[ f(x,y) =B+ A \cos(k_y y) \f]
 */
struct CosY
{
    /**
     * @brief Construct with two coefficients
     *
     * @param amp amplitude A
     * @param bamp backgroundamp B
     * @param ky  ky
     */
    CosY( double amp, double bamp, double ky):amp_(amp), bamp_(bamp),ky_(ky){}
    DG_DEVICE
    double operator()( double x, double y)const{ return bamp_+amp_*cos(y*ky_);}
  private:
    double amp_,bamp_,ky_;
};
/**
 * @brief Inverse cosh profile
 * \f[ f(x,y) =A/\cosh^2(k_x x) \f]
 */
struct InvCoshXsq
{
    /**
     * @brief Construct with two coefficients
     *
     * @param amp amplitude
     * @param kx  kx
     */
    InvCoshXsq( double amp, double kx):amp_(amp), kx_(kx){}
    DG_DEVICE
    double operator()( double x)const{ return amp_/cosh(x*kx_)/cosh(x*kx_);}
    DG_DEVICE
    double operator()( double x, double y)const{ return this->operator()(x);}
    DG_DEVICE
    double operator()( double x, double y, double z)const{ return this->operator()(x);}
  private:
    double amp_,kx_;
};
/**
 * @brief Sin prof in x-direction
 * \f[ f(x) = f(x,y) = f(x,y,z) = B + A(1 - \sin(k_xx )) \f]
 */
struct SinProfX
{
    /**
     * @brief Construct with two coefficients
     *
     * @param amp amplitude A
     * @param bamp backgroundamp B
     * @param kx  kx
     */
    SinProfX( double amp, double bamp, double kx):amp_(amp), bamp_(bamp),kx_(kx){}
    DG_DEVICE
    double operator()( double x)const{ return bamp_+amp_*(1.-sin(x*kx_));}
    DG_DEVICE
    double operator()( double x, double y)const{ return this->operator()(x);}
    DG_DEVICE
    double operator()( double x, double y, double z)const{ return this->operator()(x);}
  private:
    double amp_,bamp_,kx_;
};
/**
 * @brief Exp prof in x-direction
 * \f[ f(x) = f(x,y) = f(x,y,z) = B + A\exp(-x/L_n) \f]
 */
struct ExpProfX
{
    /**
     * @brief Construct with two coefficients
     *
     * @param amp amplitude B
     * @param bamp backgroundamp A (choose zero for constant gradient length
     * @param ln  ln
     */
    ExpProfX( double amp, double bamp, double ln):amp_(amp), bamp_(bamp),ln_(ln){}
    DG_DEVICE
    double operator()( double x)const{ return bamp_+amp_*exp(-x/ln_);}
    DG_DEVICE
    double operator()( double x, double y)const{ return this->operator()(x);}
    DG_DEVICE
    double operator()( double x, double y, double z)const{ return this->operator()(x);}
  private:
    double amp_,bamp_,ln_;
};
/**
 * @brief A linear function in x-direction
 * \f[ f(x) = f(x,y) = f(x,y,z) = ax+b \f]
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
    double operator()( double x, double y)const{ return this->operator()(x);}
    DG_DEVICE
    double operator()( double x, double y, double z)const{ return this->operator()(x);}
   private:
    double a_,b_;
};
/**
 * @brief A linear polynomial in y-direction
 * \f[ f(x,y) = f(x,y,z) = ay+b \f]
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
    double operator()( double x, double y, double z)const { return a_*y+b_;}
    DG_DEVICE
    double operator()( double x, double y)const{ return a_*y+b_;}
  private:
    double a_,b_;
};
/**
 * @brief A linear function in z-direction
 * \f[ f(x,y,z) = az+b \f]
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
    double operator()( double x, double y, double z)const{ return a_*z+b_;}
  private:
    double a_,b_;
};

/**
 * @brief Zero outside psimax and inside psimin, otherwise 1
     \f[ \begin{cases}
        1  \text{ if } \psi_{\min} < \psi < \psi_{\max}\\
        0  \text{ else}
     \end{cases}\f]
 */
struct Iris
{
    Iris( double psi_min, double psi_max ):
        m_psimin(psi_min), m_psimax(psi_max) { }
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
 * @brief Zero outside psimax, otherwise 1
     \f[ \begin{cases}
        0  \text{ if } \psi > \psi_{\max} \\
        1  \text{ else}
     \end{cases}\f]
 */
struct Pupil
{
    Pupil( double psimax):
        psimax_(psimax) { }
    double operator()(double psi)const
    {
        if( psi > psimax_) return 0.;
        return 1.;
    }
    private:
    double psimax_;
};
/**
 * @brief Psi inside psimax and psimax outside psimax
     \f[ \begin{cases}
        \psi_{\max}  \text{ if } \psi > \psi_{\max} \\
        \psi \text{ else}
     \end{cases}\f]
 */
struct PsiPupil
{
    PsiPupil(double psimax):
        psimax_(psimax){ }
    double operator()(double psi)const
    {
        if( psi > psimax_) return psimax_;
        return  psi;
    }
    private:
    double psimax_;
};
/**
 * @brief Zero up to psimax, then one
     \f[ \begin{cases}
        1  \text{ if } \psi > \psi_{\max} \\
        0  \text{ else}
     \end{cases}\f]
 */
struct Heaviside
{
    Heaviside( double psimax):
        psimax_(psimax){ }

    double operator()(double psi)const
    {
        if( psi > psimax_) return 1.;
        return 0.;
    }
    private:
    double psimax_;
};

/**
 * @brief One up to \c psimax, then a Gaussian down to zero
     \f[ \begin{cases}
 1 \text{ if } \psi < \psi_{\max}\\
 0 \text{ if } \psi > (\psi_{\max} + 4\alpha) \\
 \exp\left( - \frac{(\psi - \psi_{\max})^2}{2\alpha^2}\right), \text{ else}
 \end{cases}
   \f]
 */
struct GaussianDamping
{
    GaussianDamping( double psimax, double alpha):
        m_psimax(psimax), m_alpha(alpha) { }
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
 * @brief Step function using tanh
 * \f[ f(x) = B + 0.5 A(1+ \text{sign} \tanh((x-x_b)/\alpha ) ) \f]
 */
struct TanhProfX {
    /**
     * @brief Construct with xb, width and sign
     *
     * @param xb boundary value
     * @param width damping width \c alpha
     * @param sign sign of the Tanh, defines the damping direction
     * @param bgamp background amplitude \c B
     * @param profamp profile amplitude \c A
     */
    TanhProfX(double xb, double width, int sign =1,double bgamp = 0.,
        double profamp = 1.) :
        xb_(xb),w_(width), s_(sign),bga_(bgamp),profa_(profamp)  {}
    DG_DEVICE
    double operator() (double x)const
    {
        return profa_*0.5*(1.+s_*tanh((x-xb_)/w_))+bga_;
    }
    DG_DEVICE
    double operator()( double x, double y)const{ return this->operator()(x);}
    DG_DEVICE
    double operator()( double x, double y, double z)const{ return this->operator()(x);}
    private:
    double xb_;
    double w_;
    int s_;
    double bga_;
    double profa_;
};

/**
 * @brief Exponential \f[ f(x) = A \exp(\lambda x)\f]
 *
 * @tparam T value-type
 */
template< class T = double >
struct EXP
{
    /**
     * @brief Coefficients of \f$ A\exp(\lambda x) \f$
     *
     * @param amp Amplitude
     * @param lambda coefficient
     */
    EXP( T amp = 1., T lambda = 1.): amp_(amp), lambda_(lambda){}
    /**
     * @brief return exponential
     *
     * @param x x
     *
     * @return \f$ A\exp(\lambda x)\f$
     */
    DG_DEVICE
    T operator() ( T x) const
    {
        return amp_*exp(lambda_*x);
    }
  private:
    T amp_, lambda_;
};
/**
 * @brief natural logarithm
 * \f[ f(x) = \ln(x)\f]
 *
 * @tparam T value-type
 */
template < class T = double>
struct LN
{
    /**
     * @brief The natural logarithm
     *
     * @param x of x
     * @return  \f$ \ln(x) \f$
     */
    DG_DEVICE
    T operator() (const T& x) const
    {
        return log(x);
    }

};
/**
 * @brief Square root
 * \f[ f(x) = \sqrt{x}\f]
 *
 * @tparam T value-type
 */
template < class T = double>
struct SQRT
{
    /**
     * @brief Square root
     *
     * @param x of x
     *
     * @return sqrt(x)
     */
    DG_DEVICE
    T operator() (T x) const
    {
        return sqrt(x);
    }

};

/**
 * @brief Minmod function
 \f[ f(x_1, x_2, x_3) = \begin{cases}
         \min(x_1, x_2, x_3) \text{ for } x_1, x_2, x_3 >0 \\
         \max(x_1, x_2, x_3) \text{ for } x_1, x_2, x_3 <0 \\
         0 \text{ else}
 \end{cases}
 \f]
 *
 * might be useful for flux limiter schemes
 * @tparam T value-type
 */
template < class T>
struct MinMod
{
    /**
     * @brief Minmod of three numbers
     *
     * @param a1 a1
     * @param a2 a2
     * @param a3 a3
     *
     * @return minmod(a1, a2, a3)
     */
    DG_DEVICE
    T operator() ( T a1, T a2, T a3)const
    {
        if( a1*a2 > 0)
            if( a1*a3 > 0)
            {
                if( a1 > 0)
                    return min( a1, a2, a3, +1.);
                else
                    return min( a1, a2, a3, -1.);
            }
        return 0.;


    }
    private:
    T min( T a1, T a2, T a3, T sign)const
    {
        T temp = sign*a1;
        if( sign*a2 < temp)
            temp = sign*a2;
        if( sign*a3 < temp)
            temp = sign*a3;
        return sign*temp;

    }
};

/**
 * @brief Add a constant value
 * \f[ f(x) = x + c\f]
 *
 * @tparam T value type
 */
template <class T = double>
struct PLUS
{
    /**
     * @brief Construct
     *
     * @param value to be added
     */
    PLUS( T value): x_(value){}
    /**
     * @brief Add a constant value
     *
     * @param x  the input
     *
     * @return  x + value
     */
    DG_DEVICE
    T operator()( T x)const{ return x + x_;}
    private:
    T x_;
};

/**
 * @brief %Invert the given value
 * \f[ f(x) = 1/x \f]
 *
 * @tparam T value type
 */
template <class T = double>
struct INVERT
{
    /**
     * @brief Invert the given value
     *
     * @param x  the input
     * @return  1/x
     */
    DG_DEVICE
    T operator()( T x)const{ return 1./x;}
};

/**
 * @brief returns (positive) modulo
 * \f[ f(x) = x\mod m\f]
 *
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
    MOD( T m): x_(m){}

    /**
     * @brief Compute mod(x, value), positively defined
     *
     * @param x
     *
     * @return
     */
    DG_DEVICE
    T operator()( T x)const{
        return (fmod(x,x_) < 0 ) ? (x_ + fmod(x,x_)) : fmod(x,x_);
    }
    private:
    T x_;

};
/**
 * @brief absolute value
 * \f[ f(x) = |x|\f]
 *
 * @tparam T value type
 */
template <class T = double>
struct ABS
{
    /**
     * @brief The absolute value
     *
     * @param x of x
     *
     * @return  abs(x)
     */
    DG_DEVICE
    T operator()(T x)const{ return fabs(x);}
};
/**
 * @brief returns positive values
 \f[ f(x) = \begin{cases}
         x \text{ for } x>0 \\
         0 \text{ else}
 \end{cases}
 \f]
 *
 * @tparam T value type
 */
template <class T = double>
struct POSVALUE
{
    /**
     * @brief Returns positive values of x
     *
     * @param x of x
     *
     * @return  x*0.5*(1+sign(x))
     */
    DG_DEVICE
    T operator()( T x)const{
        if (x >= 0.0) return x;
        return 0.0;
    }
};

/**
 * @brief Return a constant
 * \f[ f(x) = c\f]
 *
 */
struct CONSTANT
{
    /**
     * @brief Construct with a value
     *
     * @param cte the constant value
     *
     */
    CONSTANT( double cte): value_(cte){}

    /**
     * @brief constant
     *
     * @param x
     *
     * @return
     */
    DG_DEVICE
    double operator()(double x)const{return value_;}
    /**
     * @brief constant
     *
     * @param x
     * @param y
     *
     * @return
     */
    DG_DEVICE
    double operator()(double x, double y)const{return value_;}
    /**
     * @brief constant
     *
     * @param x
     * @param y
     * @param z
     *
     * @return
     */
    DG_DEVICE
    double operator()(double x, double y, double z)const{return value_;}
    private:
    double value_;
};

/**
 * @brief Return one
 * \f[ f(x) = 1\f]
 *
 */
struct ONE
{
    DG_DEVICE
    double operator()(double x)const{return 1.;}
    DG_DEVICE
    double operator()(double x, double y)const{return 1.;}
    DG_DEVICE
    double operator()(double x, double y, double z)const{return 1.;}
};
/**
 * @brief Return zero
 * \f[ f(x) = 0\f]
 *
 */
struct ZERO
{
    DG_DEVICE
    double operator()(double x)const{return 0.;}
    DG_DEVICE
    double operator()(double x, double y)const{return 0.;}
    DG_DEVICE
    double operator()(double x, double y, double z)const{return 0.;}
};

/**
 * @brief Functor returning a Lamb dipole
 \f[ f(x,y) = \begin{cases} 2\lambda U J_1(\lambda r) / J_0(\gamma)\cos(\theta) \text{ for } r<R \\
         0 \text{ else}
         \end{cases}
 \f]

 with \f$ r = \sqrt{(x-x_0)^2 + (y-y_0)^2}\f$, \f$
 \theta = \arctan_2( (y-y_), (x-x_0))\f$,
 \f$J_0, J_1\f$ are
 Bessel functions of the first kind of order 0 and 1 and
 \f$\lambda = \gamma/R\f$ with \f$ \gamma = 3.83170597020751231561\f$
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
 * @brief Return a 2d vortex function
       \f[f(x,y) =\begin{cases}
       \frac{u_d}{1.2965125} \left(
       r\left(1+\frac{\beta_i^2}{g_i^2}\right)
       - R \frac{\beta_i^2}{g_i^2} \frac{J_1(g_ir/R)}{J_1(g_i)}\right)\cos(\theta) \text{ if } r < R \\
      \frac{u_d}{1.2965125} R \frac{K_1(\beta_i {r}/{R})}{K_1(\beta)} \cos(\theta) \text{ else }
      \end{cases}
      \f]

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
     * @brief Evaluate the vortex
     *
       \f[f(x,y) =\begin{cases}
       \frac{u_d}{1.2965125} \left(
       r\left(1+\frac{\beta_i^2}{g_i^2}\right)
       - R \frac{\beta_i^2}{g_i^2} \frac{J_1(g_ir/R)}{J_1(g_i)}\right)\cos(\theta) \text{ if } r < R \\
      \frac{u_d}{1.2965125} R \frac{K_1(\beta_i {r}/{R})}{K_1(\beta)} \cos(\theta) \text{ else }
      \end{cases}
      \f]
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
    }
    /**
     * @brief Evaluate the vortex modulated by a sine wave in z
     *
       \f[f(x,y,z) =\cos(k_z z)\begin{cases}
       \frac{u_d}{1.2965125} \left(
       r\left(1+\frac{\beta_i^2}{g_i^2}\right)
       - R \frac{\beta_i^2}{g_i^2} \frac{J_1(g_ir/R)}{J_1(g_i)}\right)\cos(\theta) \text{ if } r < R \\
      \frac{u_d}{1.2965125} R \frac{K_1(\beta_i {r}/{R})}{K_1(\beta)} \cos(\theta) \text{ else }
      \end{cases}
      \f]
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
* @brief Makes a random bath in the RZ plane
*
\f[f(R,Z) = A B \sum_\vec{k} \sqrt{E_k} \alpha_k \cos{\left(k \kappa_k + \theta_k \right)}
\f]
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

        thrust::random::minstd_rand generator;
        thrust::random::normal_distribution<double> ndistribution;
        thrust::random::uniform_real_distribution<double> udistribution(0.0,tpi);
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
     * @param phi phi - coordinate
     *
     * @return the above function value
     */
    double operator()(double R, double Z, double phi)const {
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
///@}
} //namespace dg

