#ifndef _DG_FUNCTORS_CUH_
#define _DG_FUNCTORS_CUH_

#include <cmath>

namespace dg
{
 
template <class T>
struct AbsMax
{
    __host__ __device__
    T operator() (const T& x, const T& y)
    {
        T absx = x>0 ? x : -x;
        T absy = y>0 ? y : -y;
        return absx > absy ? absx : absy;
    }
};

struct Gaussian
{
    Gaussian( double x0, double y0, double sigma_x, double sigma_y, double amp)
        : x00(x0), y00(y0), sigma_x(sigma_x), sigma_y(sigma_y), amplitude(amp){}
    double operator()(double x, double y)
    {
        return  amplitude*
                   exp( -((x-x00)*(x-x00)/2./sigma_x/sigma_x +
                          (y-y00)*(y-y00)/2./sigma_y/sigma_y) );
    }
  private:
    double  x00, y00, sigma_x, sigma_y, amplitude;

};

struct Lamb
{
    Lamb(  double x0, double y0, double R, double U):R_(R), U_(U), x0_(x0), y0_(y0)
    {
        gamma_ = 3.83170597020751231561;
        lambda_ = gamma_/R;
        j_ = j0( gamma_);
        //std::cout << r_ <<u_<<x0_<<y0_<<lambda_<<gamma_<<j_<<std::endl;
    }
    double operator() (double x, double y)
    {
        double radius = sqrt( (x-x0_)*(x-x0_) + (y-y0_)*(y-y0_));
        double theta = atan2( (y-y0_),(x-x0_));

        if( radius <= R_)
            return 2.*lambda_*U_*j1( lambda_*radius)/j_*cos( theta) ;
        return 0;
    }
    double enstrophy( ) { return M_PI*U_*U_*gamma_*gamma_;}
    double energy() { return 2.*M_PI*R_*R_*U_*U_;}
  private:
    double R_, U_, x0_, y0_, lambda_, gamma_, j_;
};

template< class T>
struct EXP 
{
    __host__ __device__
    T operator() (const T& x) 
    { 
        return exp(x);
    }
};
template < class T>
struct LN
{
    __host__ __device__
    T operator() (const T& x) 
    { 
        return log(x);
    }

};

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

template <class T>
struct PLUS
{
    PLUS( T value): x_(value){}
    __host__ __device__
        T operator()(const T& x){ return x + x_;}
    private:
    T x_;
};

} //namespace dg

#endif //_DG_FUNCTORS_CUH
