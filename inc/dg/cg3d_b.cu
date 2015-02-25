#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "backend/timer.cuh"
#include "backend/evaluation.cuh"
#include "backend/derivatives.cuh"
#include "cg.h"

#include "backend/typedefs.cuh"


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
    dg::Grid3d<double> g3d( 0, lx, 0, ly, 0, lz, n, Nx, Ny, Nz, bcx, dg::PER);
    dg::HVec w3d = dg::create::weights( g3d);
    dg::HVec v3d = dg::create::inv_weights( g3d);
    dg::HVec x3 = dg::evaluate( initial, g3d);
    dg::HVec b3 = dg::evaluate ( laplace_fct, g3d);
    dg::blas2::symv( w3d, b3, b3);

    dg::DVec w3d_d(w3d), v3d_d(v3d), x3_d(x3), b3_d(b3);

    dg::HMatrix A3 = dg::create::laplacianM_perp( g3d, dg::not_normed); //this function is deprecated
    dg::DMatrix A3_d(A3);
    dg::CG<dg::HVec > pcg3_host( x3, g3d.size());
    dg::CG<dg::DVec > pcg3_d( x3_d, g3d.size());
    t.tic();
    std::cout << "Number of pcg iterations "<< pcg3_d( A3_d, x3_d, b3_d, v3d_d, eps, sqrt(lz))<<std::endl;
    t.toc();
    std::cout << "... for a precision of "<< eps<<std::endl;
    std::cout << "... on the device took "<< t.diff()<<"s\n";
    t.tic();
    std::cout << "Number of pcg iterations "<< pcg3_host( A3, x3, b3, v3d, eps, sqrt(lz))<<std::endl;
    t.toc();
    std::cout << "... for a precision of "<< eps<<std::endl;
    std::cout << "... on the host took   "<< t.diff()<<"s\n";
    //compute error
    const dg::HVec solution3 = dg::evaluate ( fct, g3d);
    dg::HVec error3( solution3);
    dg::blas1::axpby( 1.,x3,-1.,error3);

    double eps3 = dg::blas2::dot(w3d , error3);
    double norm3 = dg::blas2::dot(w3d , solution3);
    std::cout << "L2 Norm of relative error is:  " <<sqrt( eps3/norm3)<<std::endl;


    return 0;
}
