#ifndef _DG_FUNCTORS_CUH_
#define _DG_FUNCTORS_CUH_

#include <cmath>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/random/normal_distribution.h>
namespace dg
{
 
///@addtogroup functions
///@{

/**
 * @brief Functor for the absolute maximum
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
    T operator() (const T& x, const T& y)
    {
        T absx = x>0 ? x : -x;
        T absy = y>0 ? y : -y;
        return absx > absy ? absx : absy;
    }
};
/**
 * @brief Functor for the absolute maximum
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
    T operator() (const T& x, const T& y)
    {
        T absx = x<0 ? -x : x;
        T absy = y<0 ? -y : y;
        return absx < absy ? absx : -absy;
    }
};

/**
 * @brief Functor returning a gaussian
 * \f[
   f(x,y) = Ae^{-(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2})} 
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
    double operator()(double x, double y)
    {
        return  amplitude*
                   exp( -((x-x00)*(x-x00)/2./sigma_x/sigma_x +
                          (y-y00)*(y-y00)/2./sigma_y/sigma_y) );
    }
    /**
     * @brief Return the value of the gaussian modulated by 
     *
     * \f[
       f(x,y,z) = A\cos(kz)e^{-(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2})} 
       \f]
     * @param x x - coordinate
     * @param y y - coordinate
     * @param y z - coordinate
     *
     * @return gaussian
     */
    double operator()(double x, double y, double z)
    {
        return  amplitude*cos(kz_*z)*
                   exp( -((x-x00)*(x-x00)/2./sigma_x/sigma_x +
                          (y-y00)*(y-y00)/2./sigma_y/sigma_y) );
    }
  private:
    double  x00, y00, sigma_x, sigma_y, amplitude, kz_;

};
struct Gaussian3d
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
    Gaussian3d( double x0, double y0, double z0, double sigma_x, double sigma_y, double sigma_z, double amp)
        : x00(x0), y00(y0), z00(z0), sigma_x(sigma_x), sigma_y(sigma_y), sigma_z(sigma_z), amplitude(amp){}
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
    double operator()(double x, double y, double z)
    {
        return  amplitude*
                   exp( -((x-x00)*(x-x00)/2./sigma_x/sigma_x +
                          (z-z00)*(z-z00)/2./sigma_z/sigma_z +
                          (y-y00)*(y-y00)/2./sigma_y/sigma_y) );
    }
  private:
    double  x00, y00, z00, sigma_x, sigma_y, sigma_z, amplitude;

};
/**
 * @brief Functor returning a gaussian in x-direction
 * \f[
   f(x,y) = Ae^{-(\frac{(x-x_0)^2}{2\sigma_x^2}) } 
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
    double operator()(double x, double y)
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
    double operator()(double x, double y)
    {
        return  amplitude*exp( -((y-y00)*(y-y00)/2./sigma_y/sigma_y) );
    }
  private:
    double  y00, sigma_y, amplitude;

};


/**
 * @brief Functor for a linear polynomial in x-direction
 * 
 * \f[ f(x,y) = a*x+b \f]
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
     
     * @return result
     */
    double operator()( double x, double y){ return a_*x+b_;}
  private:
    double a_,b_;
};
/**
 * @brief Functor for a linear polynomial in y-direction
 * 
 * \f[ f(x,y) = a*y+b) \f]
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
     * @param x x - coordianate
     * @param y y - coordianate
     
     * @return result
     */
    double operator()( double x, double y){ return a_*y+b_;}
  private:
    double a_,b_;
};

/**
 * @brief Functor returning a Lamb dipole
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
    double operator() (double x, double y)
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
     * @return enstrophy
     */
    double enstrophy( ) { return M_PI*U_*U_*gamma_*gamma_;}

    /**
     * @brief The total energy of the dipole
     *
     * Analytic formula. True for periodic and dirichlet boundary conditions.
     * @return  energy
     */
    double energy() { return 2.*M_PI*R_*R_*U_*U_;}
  private:
    double R_, U_, x0_, y0_, lambda_, gamma_, j_;
};

struct Vortex
{
    /**
     * @brief 
     *
     * @param x0 X position
     * @param y0 Y position
     * @param state mode
     * @param R characteristic radius of dipole
     * @param u_dipole u_drift/u_dipole
     * @param kz
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
    double operator()( double x, double y)
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
    double operator()( double x, double y, double z)
    {
        return this->operator()(x,y)*cos(kz_*z);
    }
    private:
    // Returns the modified Bessel function K1(x) for positive real x.
    double bessk1(double x)
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
    double bessi1(double x) 
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
    * @param Rm2 squared Number of Fourier modes in R direction
    * @param Zm2 squared Number of Fourier modes in Z direction
    * @param RZm \f[RZm= (Rm2+Zm2)^{1/2};\f]
    * @param norm normalisation factor \f[norm= (2/(Rm* Zm))^{1/2}\f]
    * @param tpi \f[tpi= 2 \pi\f]
    * @param tpi2 \f[tpi= 4 \pi^2\f]
    * @param k \f[k=(k_R^2+k_Z^2)^{1/2}\f]
    * @param sqEkvec  \f[ E_k^{1/2}\f]
    * @param unif1 uniform real random variable between [0,2 pi]
    * @param unif2 uniform real random variable between [0,2 pi]
    * @param normal1 normal random variable with mean=0 and standarddeviation=1
    * @param normal2 normal random variable with mean=0 and standarddeviation=1
    * @param normalamp \f[normalamp= (normal1^2+normal2^2)^{1/2}\f]
    * @param normalphase \f[normalphase= arctan(normal2/normal1)\f]
    */
  double Rm2, Zm2, RZm;
  double norm,tpi,tpi2;    
  std::vector<double> kvec;
  std::vector<double> sqEkvec;
  std::vector<double> unif1, unif2, normal1,normal2,normalamp,normalphase;
      /**
     * @brief Functor returning a random field in the RZ-plane or in the first RZ-plane
     *
     * @param Rm Number of Fourier modes in R direction
     * @param Zm Number of Fourier modes in Z direction
     * @param Nz Number of planes in phi direction
     * @param R_min Minimal R (in units of rho_s)
     * @param Z_min Minimal Z (in units of rho_s)
     * @param gamma exponent of the energy function \f[E_k=(k/(k+_k0)^2)^\gamma\f](typical around 30)
     * @param eddysize \f[k_0=2*\pi*eddysize/XYm \f]
     * @param amp Amplitude
     */  
  BathRZ( unsigned Rm, unsigned Zm, unsigned Nz, double R_min, double Z_min, double gamma, double eddysize, double amp) : Rm_(Rm), Zm_(Zm), Nz_(Nz),R_min_(R_min), Z_min_(Z_min), gamma_(gamma), eddysize_(eddysize) , amp_(amp) {
    std::cout << "Constructing initial bath" << "\n";
    Rm2=(double)Rm_*(double)Rm_;
    Zm2=(double)Zm_*(double)Zm_;
    RZm= sqrt(Rm2+Zm2);
    norm=sqrt(2./(double)Rm_/(double)Zm_); 
    tpi=2.*M_PI; tpi2=tpi*tpi;
    double k0= tpi*eddysize_/RZm;
    double Rmh = Rm_/2.;
    double Zmh = Zm_/2.;
    
    kvec.resize(Rm_*Zm_);
    sqEkvec.resize(Rm_*Zm_);
    unif1.resize(Rm_*Zm_);
    unif2.resize(Rm_*Zm_);
    normal1.resize(Rm_*Zm_);
    normal2.resize(Rm_*Zm_);
    normalamp.resize(Rm_*Zm_);
    normalphase.resize(Rm_*Zm_);
    
    thrust::random::minstd_rand generator;
    thrust::random::normal_distribution<float> ndistribution;
    thrust::random::uniform_real_distribution<float> udistribution(0.0,tpi);
    
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
  double operator()(double R, double Z)
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
    return amp_*norm*abs(f);    
  }
    /**
     * @brief Return the value of the Bath for first phi plane
     *
     * @param R R - coordinate
     * @param Z Z - coordinate
     * @param phi phi - coordinate
     *
     */
  double operator()(double R, double Z, double phi) { 
    double f, RZphasecos;
    double  RR, ZZ;
    RR=R-R_min_;
    ZZ=Z-Z_min_;
    f=0;
    if (phi== M_PI/Nz_)
    {
        for (unsigned j=0;j<Zm_;j++)
        {
            for (unsigned i=0;i<Rm_;i++)
            {
                int z=(j)*(Rm_)+(i);
                RZphasecos= RR*unif1[z]+ZZ*unif2[z];        
                f+= sqEkvec[z]*normalamp[z]*cos(kvec[z]*RZphasecos+normalphase[z]); 
            }      
        }
    return 1.+amp_*norm*abs(f);
    }
    else {
    return 1.;
    }
  }
  private:
  unsigned Rm_,Zm_,Nz_;
  double gamma_, eddysize_, R_min_, Z_min_;
  double amp_;
};
/**
 * @brief Exponential
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
    T operator() (const T& x) 
    { 
        return amp_*exp(lambda_*x);
    }
  private:
    double amp_, lambda_;
};
/**
 * @brief natural logarithm
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
    T operator() (const T& x) 
    { 
        return log(x);
    }

};
/**
 * @brief Square root 
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
    T operator() (const T& x) 
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
    T operator() ( T a1, T a2, T a3)
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
    T min( T a1, T a2, T a3, T sign)
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
 *
 * @tparam T value type
 */
template <class T>
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
        T operator()(const T& x){ return x + x_;}
    private:
    T x_;
};

/**
 * @brief absolute value
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
        T operator()(const T& x){ return fabs(x);}
};

///@}
} //namespace dg

#endif //_DG_FUNCTORS_CUH
