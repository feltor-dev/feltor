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
    private:
    double R_0, I_0;
};

//psi = 0.5*r^2
//b_phi = I_0/R/sqrt(I_0*I_0+r2) = I_0/R/B
//b_R =   Z/R/B
double R_0 = 150;
double I_0 = 40;
//func is a polynomial in R,Z so exact for higher order dG
double func(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z;
    double B = sqrt(I_0*I_0+r2)/R;
    double bphi = I_0/R/R/B;
    return 1/bphi/R*sin(phi);
}
double deri(double R, double Z, double phi)
{
    //double r2 = (R-R_0)*(R-R_0)+Z*Z;
    return cos(phi)/R;
}
double deri2(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z;
    double B = sqrt(I_0*I_0+r2)/R;
    double bphi = I_0/R/R/B;
    double bR = Z/R/B;
    return -bphi/R*sin(phi) - bR*cos(phi)/R/R ;
}

int main()
{
    Field field( R_0, I_0);
    std::cout << "Type n, Nx, Ny, Nz\n";
    std::cout << "Note, that function is resolved exactly in R,Z for n > 2\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;
    double z0 = 0, z1 = 2.*M_PI;
    //double z0 = M_PI/2., z1 = 3./2.*M_PI;
    dg::Grid3d<double> g3d( R_0 - 1, R_0+1, -1, 1, z0, z1,  n, Nx, Ny, Nz);
    const dg::DVec w3d = dg::create::weights( g3d);
    dg::DZ<dg::DMatrix, dg::DVec> dz( field, g3d, 1e-4, dg::DefaultLimiter());
    //dz.set_boundaries( dg::PER, 0, 0);
    dz.set_boundaries( dg::DIR, 0., -0.);

    dg::DVec function = dg::evaluate( func, g3d), derivative(function), 
             dzz(dg::evaluate(deri2, g3d));
    const dg::DVec solution = dg::evaluate( deri, g3d);
    const dg::DVec solution2 = dg::evaluate( deri2, g3d);
    dz( function, derivative);
    dz.dzz( function, dzz);
    //dz( derivative, dzz);
    dg::blas1::axpby( 1., solution, -1., derivative);
    double norm = dg::blas2::dot( w3d, solution);
    std::cout << "Norm Solution  "<<sqrt( norm)<<"\n";
    std::cout << "Relative Difference in DZ is "<< sqrt( dg::blas2::dot( derivative, w3d, derivative)/norm )<<"\n";    
    dg::blas1::axpby( 1., solution2, -1., dzz);
    norm = dg::blas2::dot( w3d, solution2);
    std::cout << "Relative Difference in DZZ is "<< sqrt( dg::blas2::dot( w3d, dzz)/norm )<<"\n";    
    return 0;
}
