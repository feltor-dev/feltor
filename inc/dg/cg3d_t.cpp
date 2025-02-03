#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "pcg.h"
#include "elliptic.h"

const unsigned n = 3; //global relative error in L2 norm is O(h^P)
const unsigned Nx = 20;  //more N means less iterations for same error
const unsigned Ny = 20;  //more N means less iterations for same error
const unsigned Nz = 4;  //more N means less iterations for same error
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
const double lz = 10.;

const double eps_ = 1e-6;

double fct(double x, double y){ return sin(y)*sin(x);}
double laplace_fct( double x, double y) { return 2*sin(y)*sin(x);}
double initial( double x, double y) {return sin(0);}
double fct(double x, double y, double z){ return sin(y)*sin(x);}
double laplace_fct( double x, double y, double z) { return 2*sin(y)*sin(x);}
double initial( double x, double y, double z) {return sin(0);}

int main()
{
    std::cout << "TEST 3D VERSION\n";
    dg::Grid3d g3d( 0, lx, 0, ly, 0, lz, n, Nx, Ny, Nz, dg::PER, dg::PER);
    dg::HVec w3d = dg::create::weights( g3d);
    dg::HVec v3d = dg::create::inv_weights( g3d);
    dg::HVec x3 = dg::evaluate( initial, g3d);

    dg::Elliptic<dg::CartesianGrid3d, dg::HMatrix, dg::HVec> A3( g3d);
    const dg::HVec b3 = dg::evaluate ( laplace_fct, g3d);
    const dg::HVec solution3 = dg::evaluate ( fct, g3d);
    dg::PCG<dg::HVec > pcg3( x3, g3d.size());
    std::cout << "Number of pcg iterations "<< pcg3.solve( A3, x3, b3, 1., w3d, eps_)<<std::endl;
    std::cout << "For a precision of "<< eps_<<std::endl;
    //compute error
    dg::HVec error3( solution3);
    dg::blas1::axpby( 1.,x3,-1.,error3);

    dg::HVec Ax3(x3), res3( b3);
    dg::blas2::symv(  A3, x3, Ax3);
    dg::blas1::axpby( 1.,Ax3,-1.,res3);

    double xnorm3 = dg::blas2::dot( w3d, x3);
    std::cout << "L2 Norm2 of x0 is              " << xnorm3 <<"\n";
    double eps3 = dg::blas2::dot(w3d , error3);
    std::cout << "L2 Norm2 of Error is           " << eps3 <<"\n";
    double norm3 = dg::blas2::dot(w3d , solution3);
    std::cout << "L2 Norm2 of Solution is        " << norm3 <<"\n";
    double normres3 = dg::blas2::dot( w3d, res3);
    std::cout << "L2 Norm2 of Residuum is        " << normres3 <<"\n";
    std::cout << "L2 Norm of relative error is   " <<sqrt( eps3/norm3)<<std::endl;

    return 0;
}
