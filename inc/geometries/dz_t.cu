#include <iostream>

#include <cusp/print.h>

#include "blas.h"
#include "ds.h"
#include "functors.h"

#include "backend/functions.h"
#include "backend/timer.cuh"
#include "geometry.h"

double sine(double x, double y, double z){return sin(z);}
double cosine(double x, double y, double z){return cos(z);}

double func(double x, double y, double z)
{
    double r2 = x*x+y*y;
    return r2*sin(z);
}
double deri(double x, double y, double z)
{
    double r2 = x*x+y*y;
    return r2*cos(z);
}
double r2( double x, double y) {return x*x+y*y;}
double r2z( double x, double y, double z) {return (x*x+y*y)*z;}


int main()
{
    std::cout << "Type n, Nx, Ny, Nz\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;
    std::cout << "You typed "<<n<<" "<<Nx<<" "<<Ny<<" "<<Nz<<std::endl;
    dg::CartesianGrid3d g3d( -1, 1, -1, 1, 0.1, M_PI+0.1, n, Nx, Ny, Nz, dg::DIR, dg::DIR, dg::NEU);
    dg::CartesianGrid2d perp_grid( -1, 1, -1, 1, n, Nx, Ny, dg::DIR, dg::DIR);
    const dg::DVec w3d = dg::create::volume( g3d);
    dg::Timer t;
    t.tic();
    dg::DDS::FieldAligned dsFA( dg::DefaultField(), g3d, 1e-10, dg::DefaultLimiter(), dg::NEU);

    dg::DDS ds ( dsFA, dg::DefaultField(), dg::not_normed, dg::centered);
    t.toc();
    std::cout << "TEST STRAIGHT FIELD LINES AND BOUNDARIES IN Z\n";
    std::cout << "Creation of parallel Derivative took     "<<t.diff()<<"s\n";

    dg::DVec function = dg::evaluate( func, g3d), derivative(function);
    dg::DVec constfunc = dg::evaluate( sine, g3d);
    const dg::DVec solution = dg::evaluate( deri, g3d);
    const dg::DVec constsolution = dg::evaluate( cosine, g3d);
    t.tic();
    ds.set_boundaries( dg::DIR, sin(g3d.z0()),sin(g3d.z1()));
    ds( constfunc, derivative);
    t.toc();
    std::cout << "Application of parallel Derivative took  "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., constsolution, -1., derivative);
    double norm = dg::blas2::dot( constsolution, w3d, constsolution);
    double diff = sqrt( dg::blas2::dot( derivative, w3d, derivative)/norm );
    std::cout << "DIR const: Relative Difference Is "<< diff<<"\n";
    t.tic();
    ds.set_boundaries( dg::NEU, cos(g3d.z0()),cos(g3d.z1()));
    ds( constfunc, derivative);
    t.toc();
    std::cout << "Application of parallel Derivative took  "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., constsolution, -1., derivative);
    diff = sqrt( dg::blas2::dot( derivative, w3d, derivative)/norm );
    std::cout << "NEU const: Relative Difference Is "<< diff << "\n";

    t.tic();
    dg::DVec left = dg::evaluate( r2, perp_grid), right(left);
    dg::blas1::scal( left, sin(g3d.z0()));
    dg::blas1::scal( right, sin(g3d.z1()));
    ds.set_boundaries( dg::DIR, left,right);
    ds( function, derivative);
    t.toc();
    std::cout << "Application of parallel Derivative took  "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., solution, -1., derivative);
    diff = sqrt( dg::blas2::dot( derivative, w3d, derivative)/norm );
    std::cout << "DIR l/r: Relative Difference Is "<< diff << "\n";
    t.tic();
    dg::DVec global = dg::evaluate( r2z, g3d);
    ds.set_boundaries( dg::DIR, global, sin(g3d.z0())/(g3d.z0()+g3d.hz()/2.), sin(g3d.z1())/(g3d.z1()-g3d.hz()/2.));
    ds( function, derivative);
    t.toc();
    std::cout << "Application of parallel Derivative took  "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., solution, -1., derivative);
    diff = sqrt( dg::blas2::dot( derivative, w3d, derivative)/norm );
    std::cout << "DIR global: Relative Difference Is "<< diff <<"\n";
    
    return 0;
}
