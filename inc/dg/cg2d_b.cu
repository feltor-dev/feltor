#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "matrix_traits_thrust.h"
#include "timer.cuh"
#include "evaluation.cuh"
#include "cg.cuh"
#include "derivatives.cuh"

#include "typedefs.cuh"


//leo3 can do 350 x 350 but not 375 x 375
const double ly = 2.*M_PI;

const double eps = 1e-6; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

const double lx = M_PI;
double fct(double x, double y){ return sin(y)*sin(x);}
double derivative( double x, double y){return cos(x)*sin(y);}
double laplace_fct( double x, double y) { return 2*sin(y)*sin(x);}
dg::bc bcx = dg::DIR;
//const double lx = 2./3.*M_PI;
//double fct(double x, double y){ return sin(y)*sin(3.*x/4.);}
//double laplace_fct( double x, double y) { return 25./16.*sin(y)*sin(3.*x/4.);}
//dg::bc bcx = dg::DIR_NEU;
double initial( double x, double y) {return sin(0);}


int main()
{
    dg::Timer t;
    unsigned n, Nx, Ny; 
    std::cout << "Type n, Nx and Ny\n";
    std::cin >> n >> Nx >> Ny;
    dg::Grid2d<double> grid( 0., lx, 0, ly, n, Nx, Ny, bcx, dg::PER);
    const dg::HVec s2d_h = dg::create::weights( grid);
    const dg::DVec s2d_d( s2d_h);
    const dg::HVec t2d_h = dg::create::precond( grid);
    const dg::DVec t2d_d( t2d_h);
    std::cout<<"Expand initial condition\n";
    dg::HVec x = dg::evaluate( initial, grid);

    std::cout << "Create symmetric Laplacian\n";
    t.tic();
    dg::DMatrix dA = dg::create::laplacianM( grid, dg::not_normed, dg::forward); 
    dg::DMatrix DX = dg::create::dx( grid);
    dg::HMatrix A = dA;
    t.toc();
    std::cout<< "Creation took "<<t.diff()<<"s\n";

    dg::CG< dg::DVec > pcg( x, n*n*Nx*Ny);
    dg::CG< dg::HVec > pcg_host( x, n*n*Nx*Ny);

    std::cout<<"Expand right hand side\n";
    const dg::HVec solution = dg::evaluate ( fct, grid);
    const dg::DVec deriv = dg::evaluate( derivative, grid);
    dg::HVec b = dg::evaluate ( laplace_fct, grid);
    //compute S b
    dg::blas2::symv( s2d_h, b, b);

    //copy data to device memory
    t.tic();
    const dg::DVec dsolution( solution);
    dg::DVec db( b), dx( x);
    dg::DVec db_(b), dx_(x);
    dg::HVec b_(b), x_(x);
    t.toc();
    std::cout << "Allocation and copy to device "<<t.diff()<<"s\n";
    //////////////////////////////////////////////////////////////////////
    std::cout << "# of polynomial coefficients: "<< n <<std::endl;
    std::cout << "# of 2d cells                 "<< Nx*Ny <<std::endl;
    
    t.tic();
    std::cout << "Number of pcg iterations "<< pcg( dA, dx, db, t2d_d, eps)<<std::endl;
    t.toc();
    std::cout << "... for a precision of "<< eps<<std::endl;
    std::cout << "... on the device took "<< t.diff()<<"s\n";
    t.tic();
    dg::cg( dA, dx_, db_, t2d_d, eps, dx_.size());
    t.toc();
    std::cout << "... with function took "<< t.diff()<<"s\n";
    t.tic();
    std::cout << "Number of pcg iterations "<< pcg_host( A, x, b, t2d_h, eps)<<std::endl;
    t.toc();
    std::cout << "... for a precision of "<< eps<<std::endl;
    std::cout << "... on the host took   "<< t.diff()<<"s\n";
    t.tic();
    dg::cg( A, x_, b_, t2d_h, eps, x_.size());
    t.toc();
    std::cout << "... with function took "<< t.diff()<<"s\n";
    dg::DVec derror( dsolution);
    dg::HVec  error(  solution);
    dg::blas1::axpby( 1.,dx,-1.,derror);
    dg::blas1::axpby( 1., x,-1., error);

    double normerr = dg::blas2::dot( s2d_d, derror);
    double norm = dg::blas2::dot( s2d_d, dsolution);
    std::cout << "L2 Norm of relative error is:               " <<sqrt( normerr/norm)<<std::endl;
    dg::blas2::gemv( DX, dsolution, derror);
    dg::blas1::axpby( 1., deriv, -1., derror);
    normerr = dg::blas2::dot( s2d_d, derror); 
    norm = dg::blas2::dot( s2d_d, deriv);
    std::cout << "L2 Norm of relative error in derivative is: " <<sqrt( normerr/norm)<<std::endl;
    //both functiona and derivative converge with order P 

    return 0;
}
