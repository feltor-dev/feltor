#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cusp/print.h>

#include "backend/timer.cuh"
#include "backend/evaluation.cuh"
#include "backend/sparseblockmat.cuh"
#include "backend/derivatives.h"
#include "backend/typedefs.cuh"
#include "backend/cusp_thrust_backend.h"

#include "cg.h"
#include "elliptic.h"


const double R_0 = 1000;
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
const double lz = 2.*M_PI;
double fct(double x, double y, double z){ return sin(x-R_0)*sin(y);}
double derivative( double x, double y, double z){return cos(x-R_0)*sin(y);}
double laplace_fct( double x, double y, double z) { return -1./x*cos(x-R_0)*sin(y) + 2.*sin(y)*sin(x-R_0);}
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;
double initial( double x, double y, double z) {return sin(0);}


int main()
{
    dg::Timer t;
    unsigned n, Nx, Ny, Nz; 
    std::cout << "Type n, Nx, Ny and Nz\n";
    std::cin >> n >> Nx >> Ny >> Nz;
    double eps;
    std::cout << "Type epsilon! \n";
    std::cin >> eps;
    dg::CylindricalGrid3d grid( R_0, R_0+lx, 0, ly, 0,lz, n, Nx, Ny,Nz, bcx, bcy, dg::PER);
    dg::DVec w3d = dg::create::volume( grid);
    dg::DVec v3d = dg::create::inv_volume( grid);
    dg::DVec x = dg::evaluate( initial, grid);

    std::cout << "TEST CYLINDRICAL LAPLACIAN\n";
    std::cout << "Create Laplacian\n";
    t.tic();
    dg::Elliptic<dg::aGeometry3d, dg::DMatrix, dg::DVec> laplace(grid, dg::not_normed, dg::centered);
    dg::DMatrix DX = dg::create::dx( grid);
    t.toc();
    std::cout<< "Creation took "<<t.diff()<<"s\n";

    dg::CG< dg::DVec > pcg( x, n*n*Nx*Ny);

    std::cout<<"Expand right hand side\n";
    const dg::DVec solution = dg::evaluate ( fct, grid);
    const dg::DVec deriv = dg::evaluate( derivative, grid);
    dg::DVec b = dg::evaluate ( laplace_fct, grid);
    //compute W b
    dg::blas2::symv( w3d, b, b);
    
    std::cout << "For a precision of "<< eps<<" ..."<<std::endl;
    x = dg::evaluate( initial, grid);
    t.tic();
    std::cout << "Number of pcg iterations "<< pcg( laplace, x, b, v3d, eps)<<std::endl;
    t.toc();
    std::cout << "... on the device took "<< t.diff()<<"s\n";
    dg::DVec  error(  solution);
    dg::blas1::axpby( 1., x,-1., error);

    double normerr = dg::blas2::dot( w3d, error);
    double norm = dg::blas2::dot( w3d, solution);
    std::cout << "L2 Norm of relative error is:               " <<sqrt( normerr/norm)<<std::endl;
    dg::blas2::gemv( DX, x, error);
    dg::blas1::axpby( 1., deriv, -1., error);
    normerr = dg::blas2::dot( w3d, error); 
    norm = dg::blas2::dot( w3d, deriv);
    std::cout << "L2 Norm of relative error in derivative is: " <<sqrt( normerr/norm)<<std::endl;
    //both function and derivative converge with order P 

    return 0;
}
