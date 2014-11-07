#include <iostream>

#include <cusp/print.h>

#include "evaluation.cuh"
#include "dz.cuh"
#include "functions.h"
#include "../blas2.h"
#include "../functors.h"
#include "interpolation.cuh"


struct Field
{
    Field( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    void operator()( const std::vector<dg::HVec>& y, std::vector<dg::HVec>& yp)
    {
        for( unsigned i=0; i<y[0].size(); i++)
        {
            double gradpsi = ((y[0][i]-R_0)*(y[0][i]-R_0) + y[1][i]*y[1][i])/I_0/I_0;
            yp[2][i] = y[0][i]*sqrt(1 + gradpsi);
            yp[0][i] = y[0][i]*y[1][i]/I_0;
            //yp[1][i] = -y[0][i]*y[0][i]/I_0 + R_0/I_0*y[0][i] ;
            yp[1][i] = y[0][i]/I_0*(R_0-y[0][i]) ;
        }
    }
    void operator()( const dg::HVec& y, dg::HVec& yp)
    {
        double gradpsi = ((y[0]-R_0)*(y[0]-R_0) + y[1]*y[1])/I_0/I_0;
        yp[2] = y[0]*sqrt(1 + gradpsi);
        yp[0] = y[0]*y[1]/I_0;
        yp[1] = y[0]/I_0*(R_0-y[0]) ;
    }
    private:
    double R_0, I_0;
};

//psi = 0.5*r^2
//b_phi = I_0/R/sqrt(I_0*I_0+r2) = I_0/R/B
//b_R =   Z/R/B
double R_0 = 10;
double I_0 = 10; //großes I macht gerade Feldlinie

double func2d(double R, double Z)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z;
    double B = sqrt(I_0*I_0+r2)/R;
    double bphi = I_0/R/R/B;
    return 1/bphi/R;
}
double func(double R, double Z, double phi)
{
    return -func2d(R,Z)*cos(phi);
}
double modulate( double R, double Z, double phi) {return -cos(phi);}
double deri2d(double R, double Z)
{
    return 0;
}
double deri(double R, double Z, double phi)
{
    //double r2 = (R-R_0)*(R-R_0)+Z*Z;
    return sin(phi)/R;
}
double deri2(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z;
    double B = sqrt(I_0*I_0+r2)/R;
    double bphi = I_0/R/R/B;
    double bR = Z/R/B;
    return bphi/R*cos(phi) - bR*sin(phi)/R/R ;
}

int main()
{
    Field field( R_0, I_0);
    std::cout << "Type n, Nx, Ny, Nz\n";
    //std::cout << "Note, that function is resolved exactly in R,Z for n > 2\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;
    double z0 = 0, z1 = 2.*M_PI;
    //double z0 = M_PI/2., z1 = 3./2.*M_PI;
    dg::Grid3d<double> g3d( R_0 - 1, R_0+1, -1, 1, z0, z1,  n, Nx, Ny, Nz);
    dg::Grid2d<double> g2d( R_0 - 1, R_0+1, -1, 1,  n, Nx, Ny);
    const dg::DVec w3d = dg::create::weights( g3d);
    const dg::DVec w2d = dg::create::weights( g2d);
    dg::DZ<dg::DMatrix, dg::DVec> dz( field, g3d, g3d.hz(), 1e-9, dg::DefaultLimiter(), dg::NEU);
    dg::Grid3d<double> g3dp( R_0 - 1, R_0+1, -1, 1, z0, z1,  n, Nx, Ny, 1);
    dg::DZ<dg::DMatrix, dg::DVec> dz2d( field, g3dp, g3d.hz(), 1e-9, dg::DefaultLimiter(), dg::NEU);
    dg::DVec boundary=dg::evaluate( dg::zero, g3d);
    dz.set_boundaries( dg::PER, 0, 0);
    //dz.set_boundaries( dg::DIR, 0., -0.);
    //dz.set_boundaries( dg::DIR, boundary, 1, 1);

    dg::DVec function = dg::evaluate( func, g3d), derivative(function), 
             dzz(dg::evaluate(deri2, g3d));
    dg::DVec function2d = dg::evaluate( func2d, g2d), derivative2d( function2d) ;
    dg::DVec follow = dz.evaluate( func2d, 0), sinz(dg::evaluate( modulate, g3d));
    dg::blas1::pointwiseDot( follow, sinz, follow);
    dg::blas1::axpby( 1., function, -1., follow);
    double diff = dg::blas2::dot( w3d, follow);
    std::cout << "Difference between function and followed evaluation: "<<diff<<"\n";
    const dg::DVec solution = dg::evaluate( deri, g3d);
    const dg::DVec solution2 = dg::evaluate( deri2, g3d);
    const dg::DVec solution2d = dg::evaluate( deri2d, g2d);
    dz( function, derivative);
    dz2d( function2d, derivative2d);
    dz.dzz( function, dzz);
    //dz( derivative, dzz);
    double norm = dg::blas2::dot( w3d, solution);
    std::cout << "Norm Solution  "<<sqrt( norm)<<"\n";
    double err =dg::blas2::dot( w3d, derivative);
    std::cout << "Norm Derivative"<<sqrt( err)<<"\n";
    dg::blas1::axpby( 1., solution, -1., derivative);
    err =dg::blas2::dot( w3d, derivative);
    std::cout << "Relative Difference in DZ is "<< sqrt( err/norm )<<"\n";    
    dg::blas1::axpby( 1., solution2, -1., dzz);
    norm = dg::blas2::dot( w3d, solution2);
    std::cout << "Relative Difference in DZZ is "<< sqrt( dg::blas2::dot( w3d, dzz)/norm )<<"\n";    
    dg::blas1::axpby( 1., solution2d, -1., derivative2d);
    std::cout << "Difference in DZ2d is "<< sqrt( dg::blas2::dot( w2d, derivative2d) )<<"\n";    
    return 0;
}
