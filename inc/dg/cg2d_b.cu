#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "backend/timer.cuh"
#include "backend/evaluation.cuh"
#include "backend/derivatives.cuh"
#include "backend/typedefs.cuh"
#include "backend/cusp_thrust_backend.h"

#include "cg.h"
#include "elliptic.h"

const double lx = M_PI;
const double ly = 2.*M_PI;

double fct(double x, double y){ return sin(y)*sin(x+M_PI/2.);}
double derivative( double x, double y){return cos(x+M_PI/2.)*sin(y);}
double laplace_fct( double x, double y) { return 2*sin(y)*sin(x+M_PI/2.);}
dg::bc bcx = dg::NEU;
//double fct(double x, double y){ return sin(x);}
//double derivative( double x, double y){return cos(x);}
//double laplace_fct( double x, double y) { return sin(x);}
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
    std::cout << "Type in eps\n";
    double eps = 1e-6; //# of pcg iterations increases very much if 
    std::cin >> eps;
    dg::Grid2d<double> grid( 0., lx, 0, ly, n, Nx, Ny, bcx, dg::PER);
    const dg::HVec s2d_h = dg::create::weights( grid);
    const dg::DVec s2d_d( s2d_h);
    const dg::HVec t2d_h = dg::create::inv_weights( grid);
    const dg::DVec t2d_d( t2d_h);
    std::cout<<"Expand initial condition\n";
    dg::HVec x = dg::evaluate( initial, grid);

    std::cout << "Create symmetric Laplacian\n";
    t.tic();
    dg::DMatrix dA = dg::create::laplacianM( grid, dg::not_normed, dg::forward); 
    dg::DMatrix DX = dg::create::dx( grid);
    dg::HMatrix A = dA;
    dg::Elliptic<dg::DMatrix, dg::DVec, dg::DVec> lap(grid, dg::not_normed, dg::centered );
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
    
    std::cout << "... for a precision of "<< eps<<std::endl;
    t.tic();
    std::cout << "Number of pcg iterations "<< pcg( dA, dx, db, t2d_d, eps)<<std::endl;
    t.toc();
    std::cout << "... on the device took "<< t.diff()<<"s\n";
    t.tic();
    dg::cg( dA, dx_, db_, t2d_d, eps, dx_.size());
    t.toc();
    std::cout << "... with function took "<< t.diff()<<"s\n";
    t.tic();
    std::cout << "Number of pcg iterations "<< pcg_host( A, x, b, t2d_h, eps)<<std::endl;
    t.toc();
    std::cout << "... on the host took   "<< t.diff()<<"s\n";
    t.tic();
    dg::cg( A, x_, b_, t2d_h, eps, x_.size());
    t.toc();
    std::cout << "... with function took "<< t.diff()<<"s\n";

    dg::blas1::scal( dx, 0);
    t.tic();
    std::cout << "Number of pcg iterations "<< pcg( lap, dx, db, t2d_d, eps)<<std::endl;
    t.toc();
    std::cout << "... on the device took "<< t.diff()<<"s\n";
    dg::DVec derror( dsolution);
    dg::HVec  error(  solution);
    dg::blas1::axpby( 1.,dx,-1.,derror);
    dg::blas1::axpby( 1., x,-1., error);

    double normerr = dg::blas2::dot( s2d_d, derror);
    double norm = dg::blas2::dot( s2d_d, dsolution);
    std::cout << "L2 Norm of relative error for symmetric is: " <<sqrt( normerr/norm)<<std::endl;
    double normerr2 = dg::blas2::dot( s2d_h, error);
    double norm2 = dg::blas2::dot( s2d_h, solution);
    std::cout << "L2 Norm of relative error is:               " <<sqrt( normerr2/norm2)<<std::endl;
    dg::blas2::gemv( DX, dx, derror);
    dg::blas1::axpby( 1., deriv, -1., derror);
    normerr = dg::blas2::dot( s2d_d, derror); 
    norm = dg::blas2::dot( s2d_d, deriv);
    std::cout << "L2 Norm of relative error in derivative is: " <<sqrt( normerr/norm)<<std::endl;
    //both functiona and derivative converge with order P 

    return 0;
}
