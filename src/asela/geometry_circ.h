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
    Iris( double R_0, double a, double b, double alpha): R_0(R_0), a_(a), b_(b), alpha_(alpha) {
    assert( a > b);}
    double operator( )(double x, double y, double z)
    {
        double r = sqrt( (x-R_0)*(x-R_0)+ y*y);
        if( r > a_) return 0.;
        if( r < b_) return 0.; 
        //return 0.25*(tanh(  ( r-a_+t_+alpha_ )/alpha_ ) + 1)  + 0.25*(tanh(  ( a_-alpha_-r)/alpha_ ) + 1);

        return 1.;
    }
  private:
    double R_0, a_, b_, alpha_;
};

struct Pupil
{
    Pupil( double R_0,double a, double b): R_0(R_0), a_(a), b_(b) { }
    double operator( )(double x, double y, double z)
    {
        double r = sqrt( (x-R_0)*(x-R_0) + y*y);
        if( r > a_) return 0.; 
        return 1.;
    }
  private:
    double R_0, a_, b_;

};
struct Damping
{
    /**
     * @brief 
     *
     * @param a outer radius
     * @param thickness ring thickness
     * @param alpha thickness of damping zone
     */
    Damping( double R_0, double a, double b, double alpha, double amp): R_0(R_0), a_(a), b_(b), alpha_(alpha), amp_(amp) {
    assert( a > b);}
    double operator( )(double x, double y, double z)
    {
        double r = sqrt( (x-R_0)*(x-R_0)+ y*y);
        if( r > a_) return 0.;
        if( r < (a_-3.*alpha_)) return 1.; 
        return 1. - exp( -( r-a_)*(r-a_)/2./alpha_/alpha_);
        //if( r > a_) return amp_;
        //if( r < (a_ - 3.*alpha_)) return 0.;
        //return amp_*exp( -(r-a_)*(r-a_)/2./alpha_/alpha_);
    }
  private:
    double R_0, a_, b_, alpha_, amp_;
};


struct Gradient
{
    Gradient( double R_0, double a, double thickness, double lnn_inner): R_0(R_0), a_(a), t_(thickness),lnN_inner( lnn_inner) { }
    double operator( )(double x, double y, double z)
    {
        double r = sqrt( (x-R_0)*(x-R_0) + y*y);
        if( r < (a_-t_)) return exp(lnN_inner*log(10)); 
        if( r < a_) return 1./t_*(r -a_ + t_ +exp(lnN_inner*log(10))*(a_ - r));
        return 1.;
    }
  private:
    double R_0, a_, t_, lnN_inner;
};

struct Field
{
    Field( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    void operator()( const std::vector<dg::HVec>& y, std::vector<dg::HVec>& yp)
    {
        for( unsigned i=0; i<y[0].size(); i++)
        {
            double gradpsi = ((y[0][i]-R_0)*(y[0][i]-R_0) + y[1][i]*y[1][i])/I_0/I_0;
            yp[0][i] = y[0][i]*y[1][i]/I_0;
            yp[1][i] = -y[0][i]/I_0*(y[0][i] - R_0) ;
            yp[2][i] = y[0][i]*sqrt(1 + gradpsi);
        }
    }
    //inverse B
    double operator()( double R, double Z, double phi)
    {
        double r2 = (R-R_0)*(R-R_0)+Z*Z;
        double norm = I_0/R_0;
        return norm*sqrt( R*R/(I_0*I_0+r2));
    }
    private:
    double R_0, I_0;
};

struct CurvatureR
{
    CurvatureR( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    double operator()( double R, double Z, double phi)
    {
        double r2 = (R-R_0)*(R-R_0)+Z*Z;
        double norm = I_0/R_0;
        return -norm*Z*R/(I_0*I_0 + r2)/sqrt(I_0*I_0+r2);
    }
    private:
    double R_0, I_0;
};

struct CurvatureZ
{
    CurvatureZ( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    double operator()( double R, double Z, double phi)
    {
        double r2 = (R-R_0)*(R-R_0)+Z*Z;
        double norm = I_0/R_0;
        return -norm*(-R*R_0 + I_0*I_0 + R_0*R_0 + Z*Z)/(I_0*I_0 + r2)/sqrt(I_0*I_0+r2) ;
    }
    private:
    double R_0, I_0;
};
struct GradLnB
{
    GradLnB( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    double operator()( double R, double Z, double phi)
    {
        double r2 = (R-R_0)*(R-R_0)+Z*Z;
        return -Z/sqrt(I_0*I_0+r2)/R ;
    }
    private:
    double R_0, I_0;
};

}//namespace eule
