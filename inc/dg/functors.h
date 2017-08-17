#pragma once

#include <cmath>
#include <vector>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/random/normal_distribution.h>
#include "blas1.h"
#include "backend/grid.h"
#include "backend/evaluation.cuh"
#include "backend/functions.h"
/*!@file
 * Functors to use in dg::evaluate or dg::blas1::transform functions
 */
namespace dg
{
 
///@addtogroup functions
///@{

/**
 * @brief Functor for the absolute maximum
 * \f[ f(x,y) = \max(|x|,|y|)\f]
 *
 * @tparam T value-type
 */
template <class T>
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
#ifdef __CUDACC__
    __host__ __device__
#endif
    T operator() (const T& x, const T& y) const
    {
        T absx = x>0 ? x : -x;
        T absy = y>0 ? y : -y;
        return absx > absy ? absx : absy;
    }
};
/**
 * @brief Functor for the absolute maximum
 * \f[ f(x,y) = \min(|x|,|y|)\f]
 *
 * @tparam T value-type
 */
template <class T>
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
#ifdef __CUDACC__
    __host__ __device__
#endif
    T operator() (const T& x, const T& y) const
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
   f(x,y) = Ae^{1 + \left(\frac{(x-x_0)^2}{\sigma_x^2} + \frac{(y-y_0)^2}{\sigma_y^2} - \right)^{-1}} 
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
 * @brief Functor returning a gaussian in x-direction
 * \f[
   f(x,y) = Ae^{-\frac{(x-x_0)^2}{2\sigma_x^2} } 
   \f]
 */
struct GaussianX
{
    /**
     * @brief Functor returning a gaussian in x 
     *
     * @param x0 x-center-coordinate
     * @param sigma_x x - variance
     * @param amp Amplitude
     */
    GaussianX( double x0, double sigma_x, double amp)
        :x00(x0), sigma_x(sigma_x), amplitude(amp){}
    /**
     * @brief Return the value of the gaussian
     *
     * \f[
       f(x,y) = Ae^{-(\frac{(x-x_0)^2}{2\sigma_x^2})} 
       \f]
     * @param x x - coordinate
     * @param y y - coordinate
     *
     * @return gaussian
     */
    double operator()(double x, double y) const
    {
        return  amplitude* exp( -((x-x00)*(x-x00)/2./sigma_x/sigma_x ));
    }
  private:
    double  x00, sigma_x, amplitude;

}; 
/**
 * @brief Functor returning a gaussian in y-direction
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
    double operator()(double x, double y) const
    {
        return  amplitude*exp( -((y-y00)*(y-y00)/2./sigma_y/sigma_y) );
    }
  private:
    double  y00, sigma_y, amplitude;

};
/**
 * @brief Functor returning a gaussian in z-direction
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
    double operator()(double x, double y, double z) const
    {
        return  amplitude*exp( -((z-z00)*(z-z00)/2./sigma_z/sigma_z) );
    }
  private:
    double  z00, sigma_z, amplitude;

};
/**
 * @brief Functor for a sin prof in x and y-direction
 * \f[ f(x,y) =B+ A sin(k_x x) sin(k_y y) \f]
 */
struct SinXSinY
{
    /**
     * @brief Construct with two coefficients
     *
     * @param amp amplitude
     * @param bamp backgroundamp
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
    double operator()( double x, double y)const{ return bamp_+amp_*sin(x*kx_)*sin(y*ky_);}
  private:
    double amp_,bamp_,kx_,ky_;
};
/**
 * @brief Functor for a sin prof in x-direction
 * \f[ f(x,y) =B+ A sin(k_x x) \f]
 */
struct SinX
{
    /**
     * @brief Construct with two coefficients
     *
     * @param amp amplitude
     * @param bamp backgroundamp
     * @param kx  kx
     */
    SinX( double amp, double bamp, double kx):amp_(amp), bamp_(bamp),kx_(kx){}
    /**
     * @brief Return profile
     *
     * @param x x - coordinate
     * @param y y - coordinate
     
     * @return \f$ f(x,y)\f$
     */
    double operator()( double x, double y)const{ return bamp_+amp_*sin(x*kx_);}
  private:
    double amp_,bamp_,kx_;
};

/**
 * @brief Functor for a sin prof in x-direction
 * \f[ f(x,y) = B + A(1-\sin(k_xx )) \f]
 */
struct SinProfX
{
    /**
     * @brief Construct with two coefficients
     *
     * @param amp amplitude
     * @param bamp backgroundamp
     * @param kx  kx
     */
    SinProfX( double amp, double bamp, double kx):amp_(amp), bamp_(bamp),kx_(kx){}
    /**
     * @brief Return profile
     *
     * @param x x - coordinate
     * @param y y - coordinate
     
     * @return \f$ f(x,y)\f$
     */
    double operator()( double x, double y)const{ return bamp_+amp_*(1.-sin(x*kx_));}
  private:
    double amp_,bamp_,kx_;
};
/**
 * @brief Functor for a exp prof in x-direction
 * \f[ f(x,y) = B + A\exp(-x/L_n) \f]
 */
struct ExpProfX
{
    /**
     * @brief Construct with two coefficients
     *
     * @param amp amplitude
     * @param bamp backgroundamp(choose zero for constant gradient length
     * @param ln  ln
     */
    ExpProfX( double amp, double bamp, double ln):amp_(amp), bamp_(bamp),ln_(ln){}
    /**
     * @brief Return linear polynomial in x 
     *
     * @param x x - coordinate
     * @param y y - coordinate
     
     * @return result
     */
    double operator()( double x, double y)const{ return bamp_+amp_*exp(-x/ln_);}
  private:
    double amp_,bamp_,ln_;
};
/**
 * @brief Functor for a linear polynomial in x-direction
 * \f[ f(x,y) = ax+b \f]
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
    /**
     * @brief Return linear polynomial in x 
     *
     * @param x x - coordinate
     * @param y y - coordinate
     * @param z z - coordinate
     
     * @return result
     */
   double operator()( double x, double y, double z)const { return a_*x+b_;}
    /**
     * @brief Return linear polynomial in x 
     *
     * @param x x - coordinate
     * @param y y - coordinate
     
     * @return result
     */
   double operator()( double x, double y)const{ return a_*x+b_;}
    /**
     * @brief Return linear polynomial in x 
     *
     * @param x x - coordinate
     
     * @return result
     */
   double operator()(double x)const{ return a_*x+b_;}
   private:
    double a_,b_;
};
/**
 * @brief Functor for a linear polynomial in y-direction
 * \f[ f(x,y) = ay+b \f]
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
    /**
     * @brief Return linear polynomial in x 
     *
     * @param x x - coordinate
     * @param y y - coordinate
     * @param z z - coordinate
     
     * @return result
     */
    double operator()( double x, double y, double z)const { return a_*y+b_;}
    /**
     * @brief Return linear polynomial in x 
     *
     * @param x x - coordinate
     * @param y y - coordinate
     
     * @return result
     */
    double operator()( double x, double y)const{ return a_*y+b_;}
  private:
    double a_,b_;
};
/**
 * @brief Functor for a linear polynomial in z-direction
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
    /**
     * @brief Return linear polynomial in x 
     *
     * @param x x - coordinate
     * @param y y - coordinate
     * @param z z - coordinate
     
     * @return result
     */
    double operator()( double x, double y, double z)const{ return a_*z+b_;}
  private:
    double a_,b_;
};


/**
 * @brief Functor for a step function using tanh
 * \f[ f(x,y) = 0.5 profamp(1+ sign \tanh((x-x_b)/width ) )+bgampg \f]
 */
struct TanhProfX {
    /**
     * @brief Construct with xb, width and sign
     *
     * @param xb boundary value
     * @param width damping width
     * @param sign sign of the Tanh, defines the damping direction
     * @param bgamp background amplitude 
     * @param profamp profile amplitude
     */
    TanhProfX(double xb, double width, int sign,double bgamp, double profamp) : xb_(xb),w_(width), s_(sign),bga_(bgamp),profa_(profamp)  {}
    /**
     * @brief Return left side step function
     *
     * @param x x - coordianate
     * @param y y - coordianate
     

     * @return result
     */
    double operator() (double x, double y)const
    {
        return profa_*0.5*(1.+s_*tanh((x-xb_)/w_))+bga_; 
    }
    private:
    double xb_;
    double w_;
    int s_;
    double bga_;
    double profa_;
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
        j_ = j0( gamma_);
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
    double operator() (double x, double y)const
    {
        double radius = sqrt( (x-x0_)*(x-x0_) + (y-y0_)*(y-y0_));
        double theta = atan2( (y-y0_),(x-x0_));

        if( radius <= R_)
            return 2.*lambda_*U_*j1( lambda_*radius)/j_*cos( theta) ;
        return 0;
    }
    /**
     * @brief The total enstrophy of the dipole
     *
     * Analytic formula. True for periodic and dirichlet boundary conditions.
     * @return enstrophy \f$ \pi U^2\gamma^2\f$ 

     */
    double enstrophy( ) { return M_PI*U_*U_*gamma_*gamma_;}

    /**
     * @brief The total energy of the dipole
     *
     * Analytic formula. True for periodic and dirichlet boundary conditions.
     * @return  energy \f$ 2\pi R^2U^2\f$ 
     */
    double energy() { return 2.*M_PI*R_*R_*U_*U_;}
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
    double operator()( double x, double y)const
    {
        double r = sqrt( (x-x0_)*(x-x0_)+(y-y0_)*(y-y0_));
        double theta = atan2( y-y0_, x-x0_);
        double beta = b_[s_];
        double norm = 1.2965125; 

        if( r/R_<=1.)
            return u_d*( 
                      r *( 1 +beta*beta/g_[s_]/g_[s_] ) 
                    - R_*  beta*beta/g_[s_]/g_[s_] *j1(g_[s_]*r/R_)/j1(g_[s_])
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
    double operator()( double x, double y, double z)const
    {
        return this->operator()(x,y)*cos(kz_*z);
    }
    private:
    // Returns the modified Bessel function K1(x) for positive real x.
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
 */
struct BathRZ{
      /**
     * @brief Functor returning a random field in the RZ-plane or in the first RZ-plane
     *
     * @param Rm Number of Fourier modes in R direction
     * @param Zm Number of Fourier modes in Z direction
     * @param Nz Number of planes in phi direction
     * @param R_min Minimal R (in units of rho_s)
     * @param Z_min Minimal Z (in units of rho_s)
     * @param gamma exponent of the energy function \f$E_k=(k/(k+k_0)^2)^\gamma\f$(typical around 30)
     * @param eddysize \f$k_0=2\pi eddysize/\sqrt{R_m^2+Z_m^2} \f$
     * @param amp Amplitude
     */  
    BathRZ( unsigned Rm, unsigned Zm, unsigned Nz, double R_min, double Z_min, double gamma, double eddysize, double amp) : 
        Rm_(Rm), Zm_(Zm), Nz_(Nz), 
        R_min_(R_min), Z_min_(Z_min), 
        gamma_(gamma), eddysize_(eddysize) , amp_(amp),
        kvec( Rm_*Zm_, 0), sqEkvec(kvec), unif1(kvec), unif2(kvec),
        normal1(kvec), normal2(kvec), normalamp(kvec), normalphase(kvec)
    {
        double Rm2=(double)(Rm_*Rm_);
        double Zm2=(double)(Zm_*Zm_);
        double RZm= sqrt(Rm2+Zm2);

        norm_=sqrt(2./(double)Rm_/(double)Zm_); 
        double tpi=2.*M_PI, tpi2=tpi*tpi;
        double k0= tpi*eddysize_/RZm;
        double Rmh = Rm_/2.;
        double Zmh = Zm_/2.;
        
        thrust::random::minstd_rand generator;
        thrust::random::normal_distribution<double> ndistribution;
        thrust::random::uniform_real_distribution<double> udistribution(0.0,tpi);
        for (unsigned j=1;j<=Zm_;j++)
        {
            double kZ2=tpi2*(j-Zmh)*(j-Zmh)/(Zm2);
            for (unsigned i=1;i<=Rm_;i++)
            {
                double kR2=tpi2*(i-Rmh)*(i-Rmh)/(Rm2);
                int z=(j-1)*(Rm_)+(i-1);
                kvec[z]= sqrt(kR2 + kZ2);  //radial k number
                sqEkvec[z]=pow(kvec[z]*4.*k0/(kvec[z]+k0)/(kvec[z]+k0),gamma_/2.); //Energie in k space with max at 1.
                unif1[z]=cos(udistribution(generator));
                unif2[z]=sin(udistribution(generator));
                normal1[z]=ndistribution(generator);
                normal2[z]=ndistribution(generator);
                normalamp[z]=sqrt(normal1[z]*normal1[z]+normal2[z]*normal2[z]);
                normalphase[z]=atan2(normal2[z],normal1[z]);
            }
        }
    
    }
      /**
     * @brief Return the value of the Bath
     *
     * @param R R - coordinate
     * @param Z Z - coordinate
     *
     */
    double operator()(double R, double Z)const
    { 
        double f, RZphasecos, RR, ZZ;
        RR=R-R_min_;
        ZZ=Z-Z_min_;
        f=0.;
        for (unsigned j=0;j<Zm_;j++)
        {
            for (unsigned i=0;i<Rm_;i++)
            {
                int z=j*Rm_+i;
                RZphasecos= RR*unif1[z]+ZZ*unif2[z];        
                f+= sqEkvec[z]*normalamp[z]*cos(kvec[z]*RZphasecos+normalphase[z]); 
            }      
        }
        return amp_*norm_*abs(f);    
    }
    /**
     * @brief Return the value of the Bath for first phi plane
     *
     * @param R R - coordinate
     * @param Z Z - coordinate
     * @param phi phi - coordinate
     *
     */
    double operator()(double R, double Z, double phi)const { 
        double f, RZphasecos;
        double  RR, ZZ;
        RR=R-R_min_;
        ZZ=Z-Z_min_;
        f=0;
//         if (phi== M_PI/Nz_)
//         {
            for (unsigned j=0;j<Zm_;j++)
            {
                for (unsigned i=0;i<Rm_;i++)
                {
                    int z=(j)*(Rm_)+(i);
                    RZphasecos= RR*unif1[z]+ZZ*unif2[z];        
                    f+= sqEkvec[z]*normalamp[z]*cos(kvec[z]*RZphasecos+normalphase[z]); 
                }      
            }
        return amp_*norm_*abs(f);
//         }
//         else {
//         return 0.;
//         }
    }
  private:
    /**
    * @param norm normalisation factor \f[norm= (2/(Rm* Zm))^{1/2}\f]
    * @param k \f[k=(k_R^2+k_Z^2)^{1/2}\f]
    * @param sqEkvec  \f[ E_k^{1/2}\f]
    * @param unif1 uniform real random variable between [0,2 pi]
    * @param unif2 uniform real random variable between [0,2 pi]
    * @param normal1 normal random variable with mean=0 and standarddeviation=1
    * @param normal2 normal random variable with mean=0 and standarddeviation=1
    * @param normalamp \f[normalamp= (normal1^2+normal2^2)^{1/2}\f]
    * @param normalphase \f[normalphase= arctan(normal2/normal1)\f]
    */
    unsigned Rm_,Zm_,Nz_;
    double R_min_, Z_min_;
    double gamma_, eddysize_;
    double amp_;
    double norm_;
    std::vector<double> kvec;
    std::vector<double> sqEkvec;
    std::vector<double> unif1, unif2, normal1,normal2,normalamp,normalphase;
};
/**
 * @brief Exponential
 * \f[ f(x) = \exp(x)\f]
 *
 * @tparam T value-type
 */
template< class T>
struct EXP 
{
    /**
     * @brief Coefficients of A*exp(lambda*x)
     *
     * @param amp Amplitude
     * @param lambda coefficient
     */
    EXP( double amp = 1., double lambda = 1.): amp_(amp), lambda_(lambda){}
    /**
     * @brief return exponential
     *
     * @param x x
     *
     * @return A*exp(lambda*x)
     */
#ifdef __CUDACC__
    __host__ __device__
#endif
    T operator() (const T& x) const
    { 
        return amp_*exp(lambda_*x);
    }
  private:
    double amp_, lambda_;
};
/**
 * @brief natural logarithm
 * \f[ f(x) = \ln(x)\f]
 *
 * @tparam T value-type
 */
template < class T>
struct LN
{
    /**
     * @brief The natural logarithm
     *
     * @param x of x 
     *
     * @return  ln(x)
     */
#ifdef __CUDACC__
    __host__ __device__
#endif
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
template < class T>
struct SQRT
{
    /**
     * @brief Square root
     *
     * @param x of x
     *
     * @return sqrt(x)
     */
#ifdef __CUDACC__
    __host__ __device__
#endif
    T operator() (const T& x) const
    { 
        return sqrt(x);
    }

};

/**
 * @brief Minmod function
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
#ifdef __CUDACC__
    __host__ __device__
#endif
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
#ifdef __CUDACC__
    __host__ __device__
#endif
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
#ifdef __CUDACC__
    __host__ __device__
#endif
        T operator()(const T& x)const{ return x + x_;}
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
#ifdef __CUDACC__
    __host__ __device__
#endif
        T operator()(const T& x)const{ return 1./x;}
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
#ifdef __CUDACC__
    __host__ __device__
#endif
        T operator()(const T& x)const{
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
template <class T>
struct ABS
{
    /**
     * @brief The absolute value
     *
     * @param x of x
     *
     * @return  abs(x)
     */
#ifdef __CUDACC__
    __host__ __device__
#endif
        T operator()(const T& x)const{ return fabs(x);}
};
/**
 * @brief returns positive values
 * \f[ f(x) = |x|\f]
 *
 * @tparam T value type
 */
template <class T>
struct POSVALUE
{
    /**
     * @brief Returns positive values of x
     *
     * @param x of x
     *
     * @return  x*0.5*(1+sign(x))
     */
#ifdef __CUDACC__
    __host__ __device__
#endif
        T operator()(const T& x)const{
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
#ifdef __CUDACC__
    __host__ __device__
#endif
    double operator()(double x)const{return value_;}
    /**
     * @brief constant
     *
     * @param x
     * @param y
     *
     * @return 
     */
#ifdef __CUDACC__
    __host__ __device__
#endif
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
#ifdef __CUDACC__
    __host__ __device__
#endif
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
#ifdef __CUDACC__
    __host__ __device__
#endif
    double operator()(double x)const{return 1.;}
#ifdef __CUDACC__
    __host__ __device__
#endif
    double operator()(double x, double y)const{return 1.;}
#ifdef __CUDACC__
    __host__ __device__
#endif
    double operator()(double x, double y, double z)const{return 1.;}
};
/**
 * @brief Return zero
 * \f[ f(x) = 0\f]
 *
 */
struct ZERO
{
#ifdef __CUDACC__
    __host__ __device__
#endif
    double operator()(double x)const{return 0.;}
#ifdef __CUDACC__
    __host__ __device__
#endif
    double operator()(double x, double y)const{return 0.;}
#ifdef __CUDACC__
    __host__ __device__
#endif
    double operator()(double x, double y, double z)const{return 0.;}
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

