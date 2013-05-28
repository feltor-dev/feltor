#include <iostream>
#include <iomanip>

#include <cusp/print.h>

#include "xspacelib.cuh"
#include "timer.cuh"
#include "cg.cuh"

const unsigned n = 3; //global relative error in L2 norm is O(h^P)
const unsigned Nx = 66;  //more N means less iterations for same error
const unsigned Ny = 66;  //more N means less iterations for same error

const double lx = M_PI;
const double ly = M_PI;
const double eps = 1e-3; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

double initial( double x, double y) {return 0.;}
double pol( double x, double y) {return 1. + sin(x)*sin(y); } //must be strictly positive
//double pol( double x, double y) {return 1.; }

double rhs( double x, double y) { return 2.*sin(x)*sin(y)*(sin(x)*sin(y)+1)-sin(x)*sin(x)*cos(y)*cos(y)-cos(x)*cos(x)*sin(y)*sin(y);}
//double rhs( double x, double y) { return 2.*sin( x)*sin(y);}
double sol(double x, double y)  { return sin( x)*sin(y);}

using namespace std;

//replace DVec with HVec and DMatrix with HMAtrix to compute on host vs device
typedef dg::DVec Vector;
typedef dg::DMatrix Matrix;
int main()
{
    dg::Timer t;
    dg::Grid<double, n> grid( 0, lx, 0, ly, Nx, Ny, dg::DIR, dg::DIR);
    dg::V2D<double, n> v2d( grid.hx(), grid.hy());
    dg::W2D<double, n> w2d( grid.hx(), grid.hy());
    //create functions A(chi) x = b
    Vector x =    dg::evaluate( initial, grid);
    Vector b =    dg::evaluate( rhs, grid);
    Vector chi =  dg::evaluate( pol, grid);
    const Vector solution = dg::evaluate( sol, grid);
    Vector error( solution);


    cout << "Create Polarisation object!\n";
    t.tic();
    dg::Polarisation2dX<double, n, Vector> pol( grid);
    t.toc();
    cout << "Creation of polarisation object took: "<<t.diff()<<"s\n";
    cout << "Create Polarisation matrix!\n";
    t.tic();
    Matrix A = pol.create( chi ); 
    t.toc();
    cout << "Creation of polarisation object took: "<<t.diff()<<"s\n";
    //dg::Matrix Ap= dg::create::laplacian( grid, false); 
    //cout << "Polarisation matrix: "<< endl;
    //cusp::print( A);
    //cout << "Laplacian    matrix: "<< endl;
    //cusp::print( Ap);
    cout << "Create conjugate gradient!\n";
    t.tic();
    dg::CG<Matrix, Vector, dg::V2D<double,n> > pcg( x, n*n*Nx*Ny);
    t.toc();
    cout << "Creation of polarisation matrix took: "<<t.diff()<<"s\n";

    cout << "# of polynomial coefficients: "<< n <<endl;
    cout << "# of 2d cells                 "<< Nx*Ny <<endl;
    //compute W b
    dg::blas2::symv( w2d, b, b);
    cudaThreadSynchronize();
    t.tic();
    std::cout << "Number of pcg iterations "<< pcg( A, x, b, v2d, eps)<<endl;
    t.toc();
    cout << "For a precision of "<< eps<<endl;
    cout << "Took "<<t.diff()<<"s\n";
    //compute error
    dg::blas1::axpby( 1.,x,-1., error);

    double eps = dg::blas2::dot( v2d, error);
    cout << "L2 Norm2 of Error is " << eps << endl;
    double norm = dg::blas2::dot( v2d, solution);
    std::cout << "L2 Norm of relative error is "<<sqrt( eps/norm)<<std::endl;

    return 0;
}

