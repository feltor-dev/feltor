#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "evaluation.cuh"
#include "cg.cuh"
#include "tensor.cuh"
#include "derivatives.cuh"
#include "typedefs.cuh"

const unsigned n = 3; //global relative error in L2 norm is O(h^P)
const unsigned Nx = 20;  //more N means less iterations for same error
const unsigned Ny = 20;  //more N means less iterations for same error
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;

const double eps_ = 1e-6; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

double fct(double x, double y){ return sin(y)*sin(x);}
double laplace_fct( double x, double y) { return 2*sin(y)*sin(x);}
double initial( double x, double y) {return sin(0);}
double fct(double x, double y, double z){ return sin(y)*sin(x);}
double laplace_fct( double x, double y, double z) { return 2*sin(y)*sin(x);}
double initial( double x, double y, double z) {return sin(0);}

using namespace std;

int main()
{
    dg::Grid2d<double> grid( 0, lx, 0, ly,n, Nx, Ny, dg::PER, dg::PER);
    //dg::S2D<double > s2d( grid);
    //dg::T2D<double > t2d( grid);
    dg::HVec s2d = dg::create::s2d( grid);
    dg::HVec t2d = dg::create::t2d( grid);
    cout<<"Expand initial condition\n";
    dg::HVec x = dg::expand( initial, grid);

    cout << "Create Laplacian\n";
    dg::HMatrix A = dg::create::laplacianM( grid, dg::not_normed, dg::LSPACE, dg::symmetric); 
    dg::CG<dg::HVec > pcg( x, n*n*Nx*Ny);
    cout<<"Expand right hand side\n";
    dg::HVec b = dg::expand ( laplace_fct, grid);
    const dg::HVec solution = dg::expand ( fct, grid);
    //////////////////////////////////////////////////////////////////////
    cout << "# of polynomial coefficients: "<< n <<endl;
    cout << "# of 2d cells                 "<< Nx*Ny <<endl;
    //compute S b
    dg::blas2::symv( s2d, b, b);
    cout << "Number of pcg iterations "<< pcg( A, x, b, t2d, eps_)<<endl;
    //std::cout << "Number of cg iterations "<< pcg( A, x, b, dg::Identity<double>(), eps)<<endl;
    cout << "For a precision of "<< eps_<<endl;
    //compute error
    dg::HVec error( solution);
    dg::blas1::axpby( 1.,x,-1.,error);

    dg::HVec Ax(x), res( b);
    dg::blas2::symv(  A, x, Ax);
    dg::blas1::axpby( 1.,Ax,-1.,res);

    double xnorm = dg::blas2::dot( s2d, x);
    cout << "L2 Norm2 of x0 is              " << xnorm << endl;
    double eps = dg::blas2::dot(s2d , error);
    cout << "L2 Norm2 of Error is           " << eps << endl;
    double norm = dg::blas2::dot(s2d , solution);
    cout << "L2 Norm2 of Solution is        " << norm << endl;
    double normres = dg::blas2::dot( s2d, res);
    cout << "L2 Norm2 of Residuum is        " << normres << endl;
    cout << "L2 Norm of relative error is   " <<sqrt( eps/norm)<<endl;
    //Fehler der Integration des Sinus ist vernachlässigbar (vgl. evaluation_t)

    return 0;
}
