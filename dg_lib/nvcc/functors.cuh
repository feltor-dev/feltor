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
    Lamb(  double x0, double y0, double R, double U):r_(R), u_(U), x0_(x0), y0_(y0)
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

        if( radius <= r_)
            return 2.*lambda_*u_*j1( lambda_*radius)/j_*cos( theta) ;
        return 0;
    }
  private:
    double r_, u_, x0_, y0_, lambda_, gamma_, j_;
};

}

#endif //_DG_FUNCTORS_CUH
