#include <iostream>

#include <cusp/print.h>

#include "blas.h"
#include "ds.h"
#include "functors.h"

#include "backend/functions.h"
#include "backend/timer.cuh"
#include "geometry.h"

struct Field
{
    Field( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double gradpsi = ((y[0]-R_0)*(y[0]-R_0) + y[1]*y[1])/I_0/I_0;
        yp[2] = y[0]*sqrt(1 + gradpsi);
        yp[0] = y[0]*y[1]/I_0;
        yp[1] = y[0]/I_0*(R_0-y[0]) ;
    }
    double operator()( double x, double y, double z) const
    {
        double gradpsi = ((x-R_0)*(x-R_0) + y*y)/I_0/I_0;
        return  x/sqrt( 1 + gradpsi)/R_0/I_0;
    }
    double error( const dg::HVec& x0, const dg::HVec& x1)
    {
        return sqrt( (x0[0]-x1[0])*(x0[0]-x1[0]) +(x0[1]-x1[1])*(x0[1]-x1[1])+(x0[2]-x1[2])*(x0[2]-x1[2]));
    }
    bool monitor( const dg::HVec& end){ 
        if ( std::isnan(end[0]) || std::isnan(end[1]) || std::isnan(end[2]) ) 
        {
            return false;
        }
        //if new integrated point outside domain
        if ((1e-5 > end[0]  ) || (1e10 < end[0])  ||(-1e10  > end[1]  ) || (1e10 < end[1])||(-1e10 > end[2]  ) || (1e10 < end[2])  )
        {
            return false;
        }
        return true;
    }
    private:
    double R_0, I_0;
};

double R_0 = 10;
double I_0 = 40;
double func(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z;
    return r2*sin(phi);
}
double deri(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z;
    return I_0/R/sqrt(I_0*I_0 + r2)* r2*cos(phi);
}


int main()
{
    Field field( R_0, I_0);
    std::cout << "Type n, Nx, Ny, Nz\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;
    std::cout << "You typed "<<n<<" "<<Nx<<" "<<Ny<<" "<<Nz<<std::endl;
    dg::CylindricalGrid3d<dg::DVec> g3d( R_0 - 1, R_0+1, -1, 1, 0, 2.*M_PI, n, Nx, Ny, Nz, dg::NEU, dg::NEU, dg::PER);
    const dg::DVec w3d = dg::create::volume( g3d);
    dg::Timer t;
    t.tic();
    dg::DDS::FieldAligned dsFA( field, g3d, 1e-10, dg::DefaultLimiter(), dg::NEU);

    dg::DDS ds ( dsFA, field, dg::not_normed, dg::centered);
    t.toc();
    std::cout << "Creation of parallel Derivative took     "<<t.diff()<<"s\n";

    dg::DVec function = dg::evaluate( func, g3d), derivative(function);
    const dg::DVec solution = dg::evaluate( deri, g3d);
    t.tic();
    ds( function, derivative);
    t.toc();
    std::cout << "Application of parallel Derivative took  "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., solution, -1., derivative);
    double norm = dg::blas2::dot( w3d, solution);
    std::cout << "Norm Solution "<<sqrt( norm)<<"\n";
    std::cout << "Relative Difference Is "<< sqrt( dg::blas2::dot( derivative, w3d, derivative)/norm )<<"\n";
    std::cout << "Error is from the parallel derivative only if n>2\n"; //since the function is a parabola
    dg::Gaussian init0(R_0+0.5, 0, 0.2, 0.2, 1);
    dg::GaussianZ modulate(0., M_PI/3., 1);
    t.tic();
    function = ds.fieldaligned().evaluate( init0, modulate, Nz/2, 2);
    t.toc();
    std::cout << "Fieldaligned initialization took "<<t.diff()<<"s\n";
    ds( function, derivative);
    norm = dg::blas2::dot(w3d, derivative);
    std::cout << "Norm Centered Derivative "<<sqrt( norm)<<" (compare with that of ds_mpib)\n";
    ds.forward( function, derivative);
    norm = dg::blas2::dot(w3d, derivative);
    std::cout << "Norm Forward  Derivative "<<sqrt( norm)<<" (compare with that of ds_mpib)\n";
    ds.backward( function, derivative);
    norm = dg::blas2::dot(w3d, derivative);
    std::cout << "Norm Backward Derivative "<<sqrt( norm)<<" (compare with that of ds_mpib)\n";
    
    return 0;
}
