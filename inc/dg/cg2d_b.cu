#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "timer.cuh"
#include "cusp_eigen.h"
#include "evaluation.cuh"
#include "cg.cuh"
#include "derivatives.cuh"
#include "preconditioner.cuh"

#include "typedefs.cuh"


//leo3 can do 350 x 350 but not 375 x 375
const double ly = 2.*M_PI;

const double eps = 1e-6; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

typedef dg::T2D<double> Preconditioner;
typedef dg::S2D<double> Postconditioner;

const double lx = 2.*M_PI;
const double lz = 1.;
double fct(double x, double y){ return sin(y)*sin(x);}
double laplace_fct( double x, double y) { return 2*sin(y)*sin(x);}
dg::bc bcx = dg::DIR;
//const double lx = 2./3.*M_PI;
//double fct(double x, double y){ return sin(y)*sin(3.*x/4.);}
//double laplace_fct( double x, double y) { return 25./16.*sin(y)*sin(3.*x/4.);}
//dg::bc bcx = dg::DIR_NEU;
double initial( double x, double y) {return sin(0);}

double fct(double x, double y, double z){ return sin(y)*sin(x);}
double laplace_fct( double x, double y, double z) { return 2*sin(y)*sin(x);}
//const double lx = 2./3.*M_PI;
//double fct(double x, double y){ return sin(y)*sin(3.*x/4.);}
//double laplace_fct( double x, double y) { return 25./16.*sin(y)*sin(3.*x/4.);}
//dg::bc bcx = dg::DIR_NEU;
double initial( double x, double y, double z) {return sin(0);}

using namespace std;

int main()
{
    dg::Timer t;
    unsigned n, Nx, Ny, Nz; 
    std::cout << "Type n, Nx, Ny and Nz\n";
    std::cin >> n >> Nx >> Ny>> Nz;
    dg::Grid2d<double> grid( 0, lx, 0, ly, n, Nx, Ny, bcx, dg::PER);
    dg::S2D<double> s2d( grid);
    cout<<"Expand initial condition\n";
    dg::HVec x = dg::expand( initial, grid);

    cout << "Create Laplacian\n";
    t.tic();
    dg::DMatrix dA = dg::create::laplacianM( grid, dg::not_normed, dg::LSPACE, dg::symmetric); 
    dg::HMatrix A = dA;
    t.toc();
    cout<< "Creation took "<<t.diff()<<"s\n";

    //create conjugate gradient and one eigen Cholesky
    dg::CG< dg::DVec > pcg( x, n*n*Nx*Ny);
    dg::CG< dg::HVec > pcg_host( x, n*n*Nx*Ny);
    //dg::SimplicialCholesky sol;
    //sol.compute( A);

    cout<<"Expand right hand side\n";
    const dg::HVec solution = dg::expand ( fct, grid);
    dg::HVec b = dg::expand ( laplace_fct, grid);
    //compute S b
    dg::blas2::symv( s2d, b, b);

    //copy data to device memory
    t.tic();
    const dg::DVec dsolution( solution);
    dg::DVec db( b), dx( x);
    t.toc();
    cout << "Allocation and copy to device "<<t.diff()<<"s\n";
    //////////////////////////////////////////////////////////////////////
    cout << "# of polynomial coefficients: "<< n <<endl;
    cout << "# of 2d cells                 "<< Nx*Ny <<endl;
    
    t.tic();
    cout << "Number of pcg iterations "<< pcg( dA, dx, db, Preconditioner(grid), eps)<<endl;
    t.toc();
    cout << "... for a precision of "<< eps<<endl;
    cout << "... on the device took "<< t.diff()<<"s\n";
    t.tic();
    cout << "Number of pcg iterations "<< pcg_host( A, x, b, Preconditioner(grid), eps)<<endl;
    t.toc();
    cout << "... for a precision of "<< eps<<endl;
    cout << "... on the host took   "<< t.diff()<<"s\n";
    //t.tic();
    //cout << "Success (1) "<< sol.solve( x.data().data(), b.data().data(), n*n*Nx*Ny)<<endl;
    //t.toc();
    //cout << "Cholesky took          "<< t.diff()<<"s\n";
    //compute error
    dg::DVec derror( dsolution);
    dg::HVec  error(  solution);
    dg::blas1::axpby( 1.,dx,-1.,derror);
    dg::blas1::axpby( 1., x,-1., error);

    double normerr = dg::blas2::dot( s2d, derror);
    //cout << "L2 Norm2 of CG Error is        " << normerr << endl;
    //double normerr2= dg::blas2::dot( s2d,  error);
    //cout << "L2 Norm2 of Cholesky Error is  " << normerr2 << endl;
    double norm = dg::blas2::dot( s2d, dsolution);
    cout << "L2 Norm of relative error is   " <<sqrt( normerr/norm)<<endl;
    //cout << "L2 Norm of relative error is   " <<sqrt( normerr2/norm)<<endl;

    cout << "TEST 3D VERSION\n";
    dg::Grid3d<double> g3d( 0, lx, 0, ly, 0, lz, n, Nx, Ny, Nz, bcx, dg::PER);
    dg::HVec w3d = dg::create::w3d( g3d);
    dg::HVec v3d = dg::create::v3d( g3d);
    dg::HVec x3 = dg::evaluate( initial, g3d);
    dg::HVec b3 = dg::evaluate ( laplace_fct, g3d);
    dg::blas2::symv( w3d, b3, b3);

    dg::DVec w3d_d(w3d), v3d_d(v3d), x3_d(x3), b3_d(b3);

    dg::HMatrix A3 = dg::create::laplacianM_perp( g3d, dg::not_normed, dg::XSPACE); 
    dg::DMatrix A3_d(A3);
    dg::CG<dg::HVec > pcg3_host( x3, g3d.size());
    dg::CG<dg::DVec > pcg3_d( x3_d, g3d.size());
    t.tic();
    cout << "Number of pcg iterations "<< pcg3_d( A3_d, x3_d, b3_d, v3d_d, eps)<<endl;
    t.toc();
    cout << "... for a precision of "<< eps<<endl;
    cout << "... on the device took "<< t.diff()<<"s\n";
    t.tic();
    cout << "Number of pcg iterations "<< pcg3_host( A3, x3, b3, v3d, eps)<<endl;
    t.toc();
    cout << "... for a precision of "<< eps<<endl;
    cout << "... on the host took   "<< t.diff()<<"s\n";
    //compute error
    const dg::HVec solution3 = dg::evaluate ( fct, g3d);
    dg::HVec error3( solution3);
    dg::blas1::axpby( 1.,x3,-1.,error3);

    double eps3 = dg::blas2::dot(w3d , error3);
    double norm3 = dg::blas2::dot(w3d , solution3);
    //cout << "L2 Norm2 of Solution is        " << norm3 << endl;
    cout << "L2 Norm of relative error is   " <<sqrt( eps3/norm3)<<endl;

    return 0;
}
