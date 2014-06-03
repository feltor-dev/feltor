#pragma once
#include <cassert>

namespace eule
{
struct Iris
{
    /**
     * @brief 
     *
     * @param a outer radius
     * @param thickness ring thickness
     * @param alpha thickness of damping zone
     */
    Iris( double a, double thickness, double alpha): a_(a), t_(thickness), alpha_(alpha) {
    assert( a > thickness);}
    double operator( )(double x, double y, double z)
    {
        double r = sqrt( x*x + y*y);
        if( r > a_) return 0.;
        if( r < (a_-t_)) return 0.; 
        return 0.25*(tanh(  ( r-a_+t_+alpha_ )/alpha_ ) + 1)  + 0.25*(tanh(  ( a_-alpha_-r)/alpha_ ) + 1);

        return 1.;
    }
  private:
    double a_, t_, alpha_;
};

struct Pupil
{
    Pupil( double a, double thickness): a_(a), t_(thickness) { }
    double operator( )(double x, double y, double z)
    {
        double r = sqrt( x*x + y*y);
        if( r < (a_-t_)) return 1.; 
        return 0.;
    }
  private:
    double a_, t_;

};

struct Gradient
{
    Gradient( double a, double thickness, double lnn_inner): a_(a), t_(thickness),lnN_inner( lnn_inner) { }
    double operator( )(double x, double y, double z)
    {
        double r = sqrt( x*x + y*y);
        if( r < (a_-t_)) return exp(lnN_inner*log(10)); 
        if( r < a_) return 1./t_*(r -a_ + t_ +exp(lnN_inner*log(10))*(a_ - r));
        return 1.;
    }
  private:
    double a_, t_, lnN_inner;
};

}//namespace eule
