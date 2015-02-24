#include <iostream>
#include <iomanip>

#include "timer.cuh"
#include <cusp/copy.h>
#include <cusp/print.h>
#include <cusp/hyb_matrix.h>

#include "xspacelib.cuh"
#include "polarisation.cuh"
#include "cg.h"


//NOTE: IF DEVICE=CPU THEN THE POLARISATION ASSEMBLY IS NOT PARALLEL AS IT IS NOW 

//global relative error in L2 norm is O(h^P)
//as a rule of thumb with n=4 the true error is err = 1e-3 * eps as long as eps > 1e3*err

const double lx = M_PI/2.;
const double ly = M_PI;
dg::bc bcx = dg::DIR_NEU;
//const double eps = 1e-3; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

double initial( double x, double y) {return 0.;}
double amp = 1;
//double pol( double x, double y) {return 1. + amp*sin(x)*sin(y); } //must be strictly positive
//double pol( double x, double y) {return 1.; }
double pol( double x, double y) {return 1. + sin(x)*sin(y) + x; } //must be strictly positive

//double rhs( double x, double y) { return 2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);}
//double rhs( double x, double y) { return 2.*sin( x)*sin(y);}
double rhs( double x, double y) { return 2.*sin(x)*sin(y)*(sin(x)*sin(y)+1)-sin(x)*sin(x)*cos(y)*cos(y)-cos(x)*cos(x)*sin(y)*sin(y)+(x*sin(x)-cos(x))*sin(y) + x*sin(x)*sin(y);}
double sol(double x, double y)  { return sin( x)*sin(y);}
double der(double x, double y)  { return cos( x)*sin(y);}

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
    dg::Grid2d<double> grid( 0, lx, 0, ly, n, Nx, Ny, bcx, dg::DIR);
    Vector v2d = dg::create::inv_weights( grid);
    Vector w2d = dg::create::weights( grid);
    //create functions A(chi) x = b
    Vector x =    dg::evaluate( initial, grid);
    Vector b =    dg::evaluate( rhs, grid);
    Vector chi =  dg::evaluate( pol, grid);


    cout << "Create HOST Polarisation object!\n";
    t.tic();
    dg::Polarisation2dX<dg::HVec> pol_host( grid, dg::backward);
    t.toc();
    cout << "Creation of HOST polarisation object took: "<<t.diff()<<"s\n";
    cout << "Create DEVICE Polarisation object!\n";
    t.tic();
    dg::Polarisation2dX<dg::DVec, dg::DMatrix> pol_device( grid, dg::backward);
    t.toc();
    cout << "Creation of DEVICE polarisation object took: "<<t.diff()<<"s\n";
    cout << "Create Polarisation matrix!\n";
    dg::Timer ti;
    ti.tic();
    t.tic();
    dg::HMatrix B_ = pol_host.create(chi);
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
    ti.toc();
    std::cout <<"TOTAL TIME: "<<ti.diff()<<"s\n";


    cout << "# of polynomial coefficients: "<< n <<endl;
    cout << "# of 2d cells                 "<< Nx*Ny <<endl;
    dg::CG<Vector > pcg( x, n*n*Nx*Ny);
    //compute W b
    dg::blas2::symv( w2d, b, b);
    t.tic();
    std::cout << "Number of pcg iterations "<< pcg( A, x, b, v2d, eps)<<endl;
    t.toc();
    cout << "For a precision of "<< eps<<endl;
    cout << "Took "<<t.diff()<<"s\n";

    //compute error
    const Vector solution = dg::evaluate( sol, grid);
    const Vector derivati = dg::evaluate( der, grid);
    Vector error( solution);
    dg::blas1::axpby( 1.,x,-1., error);

    double err = dg::blas2::dot( w2d, error);
    std::cout << "L2 Norm2 of Error is " << err << endl;
    double norm = dg::blas2::dot( w2d, solution);
    std::cout << "L2 Norm of relative error is "<<sqrt( err/norm)<<std::endl;
    Matrix DX = dg::create::dx( grid);
    dg::blas2::gemv( DX, x, error);
    dg::blas1::axpby( 1.,derivati,-1., error);
    err = dg::blas2::dot( w2d, error);
    std::cout << "L2 Norm2 of Error in derivative is " << err << endl;
    norm = dg::blas2::dot( w2d, derivati);
    std::cout << "L2 Norm of relative error in derivative is "<<sqrt( err/norm)<<std::endl;
    //derivative converges with p-1, for p = 1 with 1/2
    return 0;
}

