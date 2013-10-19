#include <iostream>
#include <iomanip>

#include "timer.cuh"
#include <cusp/copy.h>
#include <cusp/print.h>
#include <cusp/hyb_matrix.h>

#include "xspacelib.cuh"
#include "cg.cuh"

//global relative error in L2 norm is O(h^P)
//as a rule of thumb with n=4 the true error is err = 1e-3 * eps as long as eps > 1e3*err

const double lx = M_PI;
const double ly = 2.*M_PI;
//const double eps = 1e-3; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

double initial( double x, double y) {return 0.;}
double amp = 1;
double pol( double x, double y) {return 1. + amp*sin(x)*sin(y); } //must be strictly positive
//double pol( double x, double y) {return 1.; }

double rhs( double x, double y) { return 2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);}
//double rhs( double x, double y) { return 2.*sin( x)*sin(y);}
double sol(double x, double y)  { return sin( x)*sin(y);}

using namespace std;

//replace DVec with HVec and DMatrix with HMAtrix to compute on host vs device
typedef dg::DVec Vector;
typedef dg::DMatrix Matrix;
//typedef cusp::ell_matrix<int, double, cusp::device_memory> Matrix;
//typedef dg::HVec Vector;
//typedef dg::HMatrix Matrix;
int main()
{
    dg::Timer t;
    unsigned n, Nx, Ny; 
    double eps;
    cout << "Type n, Nx and Ny and epsilon! \n";
    cin >> n >> Nx >> Ny; //more N means less iterations for same error
    cin >> eps;
    dg::Grid<double> grid( 0, lx, 0, ly, n, Nx, Ny, dg::DIR, dg::PER);
    Vector v2d = dg::create::v2d( grid);
    Vector w2d = dg::create::w2d( grid);
    //create functions A(chi) x = b
    Vector x =    dg::evaluate( initial, grid);
    Vector b =    dg::evaluate( rhs, grid);
    Vector chi =  dg::evaluate( pol, grid);
    const Vector solution = dg::evaluate( sol, grid);
    Vector error( solution);


    cout << "Create Polarisation object!\n";
    t.tic();
    dg::Polarisation2dX<dg::HVec> pol( grid);
    t.toc();
    cout << "Creation of polarisation object took: "<<t.diff()<<"s\n";
    cout << "Create Polarisation matrix!\n";
    t.tic();
    dg::HMatrix B_ = pol.create(chi);
    //Matrix A = pol.create(chi);
    t.toc();
    cout << "Creation of polarisation matrix took: "<<t.diff()<<"s\n";
    t.tic();
    //TODO: Umwandlung Memory-technisch überprüfen!!!
    //cusp::csr_matrix<int, double, cusp::device_memory> B = B_;
    cusp::ell_matrix<int, double, cusp::host_memory> B = B_;
    t.toc();
    cout << "Conversion (1) to device matrix took: "<<t.diff()<<"s\n";
    t.tic();
    Matrix A = B;  
    t.toc();
    cout << "Conversion (2) to device matrix took: "<<t.diff()<<"s\n";
    std::cout << "# of points in matrix is: "<< A.num_entries<< "\n";
    //dg::Matrix Ap= dg::create::laplacian( grid, dg::not_normed); 
    //cout << "Polarisation matrix: "<< endl;
    //cusp::print( A);
    //cout << "Laplacian    matrix: "<< endl;
    //cusp::print( Ap);
    cout << "Create conjugate gradient!\n";
    t.tic();
    dg::CG<Vector > pcg( x, n*n*Nx*Ny);
    t.toc();
    cout << "Creation of conjugate gradient took: "<<t.diff()<<"s\n";

    cout << "# of polynomial coefficients: "<< n <<endl;
    cout << "# of 2d cells                 "<< Nx*Ny <<endl;
    //compute W b
    dg::blas2::symv( w2d, b, b);
    t.tic();
    std::cout << "Number of pcg iterations "<< pcg( A, x, b, v2d, eps)<<endl;
    t.toc();
    cout << "For a precision of "<< eps<<endl;
    cout << "Took "<<t.diff()<<"s\n";
    //compute error
    dg::blas1::axpby( 1.,x,-1., error);

    double err = dg::blas2::dot( v2d, error);
    cout << "L2 Norm2 of Error is " << err << endl;
    double norm = dg::blas2::dot( v2d, solution);
    std::cout << "L2 Norm of relative error is "<<sqrt( err/norm)<<std::endl;

    return 0;
}

