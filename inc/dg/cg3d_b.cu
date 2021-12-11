#include <iostream>
#include <iomanip>

#include "pcg.h"
#include "elliptic.h"
#include "backend/timer.h"


const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
const double lz = 1.;

dg::bc bcx = dg::DIR;
double initial( double x, double y, double z) {return sin(0);}
double fct(double x, double y, double z){ return sin(y)*sin(x)*sin(2.*M_PI*z);}
double laplace_fct( double x, double y, double z) { return 2*sin(y)*sin(x)*sin(2.*M_PI*z);}

//const double lx = 2./3.*M_PI;
//double fct(double x, double y){ return sin(y)*sin(3.*x/4.);}
//double laplace_fct( double x, double y) { return 25./16.*sin(y)*sin(3.*x/4.);}
//dg::bc bcx = dg::DIR_NEU;

int main()
{
    dg::Timer t;
    unsigned n, Nx, Ny, Nz;
    std::cout << "Type n, Nx, Ny and Nz\n";
    std::cin >> n >> Nx >> Ny>> Nz;
    std::cout << "Type in eps\n";
    double eps = 1e-6;
    std::cin >> eps;

    std::cout << "TEST 3D VERSION\n";
    dg::Grid3d g3d( 0, lx, 0, ly, 0, lz, n, Nx, Ny, Nz, bcx, dg::PER);
    const dg::DVec w3d = dg::create::weights( g3d);
    dg::DVec x3 = dg::evaluate( initial, g3d);
    dg::DVec b3 = dg::evaluate ( laplace_fct, g3d);

    dg::Elliptic<dg::CartesianGrid3d, dg::DMatrix, dg::DVec> lap(g3d, dg::forward );
    dg::PCG<dg::DVec > pcg( x3, g3d.size());
    t.tic();
    std::cout << "Number of pcg iterations "<< pcg.solve( lap, x3, b3, 1., w3d, eps, sqrt(lz))<<std::endl;
    t.toc();
    std::cout << "... for a precision of "<< eps<<std::endl;
    std::cout << "... on the device took "<< t.diff()<<"s\n";
    t.tic();
    //compute error
    const dg::DVec solution3 = dg::evaluate ( fct, g3d);
    dg::DVec error3( solution3);
    dg::blas1::axpby( 1.,x3,-1.,error3);

    double eps3 = dg::blas2::dot(w3d , error3);
    double norm3 = dg::blas2::dot(w3d , solution3);
    std::cout << "L2 Norm of relative error is:  " <<sqrt( eps3/norm3)<<std::endl;


    return 0;
}
