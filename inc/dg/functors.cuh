#ifndef _DG_FUNCTORS_CUH_
#define _DG_FUNCTORS_CUH_

#include <cmath>

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
     * @brief Return the absolute maximum
     *
     * @param x left value
     * @param y right value
     *
     * @return absolute maximum
     */
    __host__ __device__
    T operator() (const T& x, const T& y)
    {
        T absx = x>0 ? x : -x;
        T absy = y>0 ? y : -y;
        return absx > absy ? absx : absy;
    }
};

/**
 * @brief Functor returning a gaussian
 * \f[
   f(x,y) = Ae^{-(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2}} 
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
     */
    Gaussian( double x0, double y0, double sigma_x, double sigma_y, double amp)
        : x00(x0), y00(y0), sigma_x(sigma_x), sigma_y(sigma_y), amplitude(amp){}
    /**
     * @brief Return the value of the gaussian
     *
     * \f[
       f(x,y) = Ae^{-(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2}} 
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
  private:
    double  x00, y00, sigma_x, sigma_y, amplitude;

};


/**
 * @brief Functor for a linear polynomial in x-direction
 * 
 * \f[ f(x,y) = a*x+b) \f]
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
     * @param x x - coordianate
     * @param y y - coordianate
     
     * @return result
     */
    double operator()( double x, double y){ return a_*x+b_;}
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

/**
 * @brief Exponential
 *
 * @tparam T value-type
 */
template< class T>
struct EXP 
{
    __host__ __device__
    T operator() (const T& x) 
    { 
        return exp(x);
    }
};
/**
 * @brief natural logarithm
 *
 * @tparam T value-type
 */
template < class T>
struct LN
{
    __host__ __device__
    T operator() (const T& x) 
    { 
        return log(x);
    }

};

/**
 * @brief Minmod function
 *
 * @tparam T value-type
 */
template < class T>
struct MinMod
{
    __host__ __device__
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
    __host__ __device__
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
    __host__ __device__
        T operator()(const T& x){ return x + x_;}
    private:
    T x_;
};




///@}
} //namespace dg

#endif //_DG_FUNCTORS_CUH
