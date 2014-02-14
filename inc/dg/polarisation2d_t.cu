#include <iostream>
#include <iomanip>

#include <cusp/print.h>

#include "xspacelib.cuh"
#include "cg.cuh"

const unsigned n = 1; //global relative error in L2 norm is O(h^P)
const unsigned Nx = 30;  //more N means less iterations for same error
const unsigned Ny = 40;  //more N means less iterations for same error

const double lx = M_PI;
const double ly = M_PI;
const double eps = 1e-3; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

double initial( double x, double y) {return 0.;}
double pol( double x, double y) {return 1. + sin(x)*sin(y); } //must be strictly positive
//double pol( double x, double y) {return 1.; }
double sol(double x, double y)  { return sin( x)*sin(y);}

double rhs( double x, double y) { return 2.*sin(x)*sin(y)*(sin(x)*sin(y)+1)-sin(x)*sin(x)*cos(y)*cos(y)-cos(x)*cos(x)*sin(y)*sin(y);}
//double rhs( double x, double y) { return 4.*sol(x,y)*sol(x,y) + 2.*sol(x,y);}
//double rhs( double x, double y) { return 2.*sin( x)*sin(y);}

using namespace std;

int main()
{
    dg::Grid<double> grid( 0, lx, 0, ly, n, Nx, Ny, dg::DIR, dg::DIR);
    dg::DVec v2d = dg::create::v2d( grid);
    dg::DVec w2d = dg::create::w2d( grid);
    //create functions A(chi) x = b
    dg::DVec x =    dg::evaluate( initial, grid);
    dg::DVec b =    dg::evaluate( rhs, grid);
    dg::DVec chi =  dg::evaluate( pol, grid);
    const dg::DVec solution = dg::evaluate( sol, grid);
    dg::DVec error( solution);

    cout << "Create Polarisation object!\n";
    dg::Polarisation2dX<dg::HVec> pol( grid);
    cout << "Create Polarisation matrix!\n";
    dg::DMatrix A = pol.create( chi ); 
    dg::Matrix Ap= dg::create::laplacianM( grid, dg::not_normed); 
    //cout << "Polarisation matrix: "<< endl;
    //cusp::print( A);
    //cout << "Laplacian    matrix: "<< endl;
    //cusp::print( Ap);
    cout << "Create conjugate gradient!\n";
    dg::CG< dg::DVec> pcg( x, n*n*Nx*Ny);

    cout << "# of polynomial coefficients: "<< n <<endl;
    cout << "# of 2d cells                 "<< Nx*Ny <<endl;
    //compute W b
    dg::blas2::symv( w2d, b, b);
    std::cout << "Number of pcg iterations "<< pcg( A, x, b, v2d, eps)<<endl;
    cout << "For a precision of "<< eps<<endl;
    //compute error
    dg::blas1::axpby( 1.,x,-1., error);

    double eps = dg::blas2::dot( w2d, error);
    cout << "L2 Norm2 of Error is " << eps << endl;
    double norm = dg::blas2::dot( w2d, solution);
    std::cout << "L2 Norm of relative error is "<<sqrt( eps/norm)<<std::endl;

    return 0;
}

