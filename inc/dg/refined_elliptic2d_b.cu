#include <iostream>
#include <iomanip>

//#include "backend/xspacelib.cuh"
#include <thrust/device_vector.h>
#include "blas.h"


#include "refined_elliptic.h"
#include "cg.h"
#include "backend/timer.cuh"

//global relative error in L2 norm is O(h^P)
//as a rule of thumb with n=4 the true error is err = 1e-3 * eps as long as eps > 1e3*err

const double lx = M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;
//const double eps = 1e-3; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

double initial( double x, double y) {return 0.;}
double amp = 0.5;
double pol( double x, double y) {return 1. + amp*sin(x)*sin(y); } //must be strictly positive
//double pol( double x, double y) {return 1.; }
//double pol( double x, double y) {return 1. + sin(x)*sin(y) + x; } //must be strictly positive

double rhs( double x, double y) { return 2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);}
//double rhs( double x, double y) { return 2.*sin( x)*sin(y);}
//double rhs( double x, double y) { return 2.*sin(x)*sin(y)*(sin(x)*sin(y)+1)-sin(x)*sin(x)*cos(y)*cos(y)-cos(x)*cos(x)*sin(y)*sin(y)+(x*sin(x)-cos(x))*sin(y) + x*sin(x)*sin(y);}
double sol(double x, double y)  { return sin( x)*sin(y);}
double der(double x, double y)  { return cos( x)*sin(y);}


int main()
{
    dg::Timer t;
    unsigned n, Nx, Ny; 
    unsigned n_ref, multiple_x, multiple_y; 
    double eps;
    std::cout << "Type n, Nx and Ny and epsilon! \n";
    std::cin >> n >> Nx >> Ny; //more N means less iterations for same error
    std::cin >> eps;
    std::cout << "Type n_refined, multiple x and multiple y! \n";
    std::cin >> n_ref >> multiple_x >> multiple_y; //more N means less iterations for same error
    std::cout << "Computation on: "<< n <<" x "<<Nx<<" x "<<Ny<<std::endl;
    //std::cout << "# of 2d cells                 "<< Nx*Ny <<std::endl;
    dg::LinearRefinement equiX( multiple_x), equiY(multiple_y);
    dg::IdentityRefinement id;
    dg::CartesianRefinedGrid2d grid_fine( equiX, equiY, 0, lx, 0, ly, n_ref, Nx, Ny, bcx, bcy);
    dg::CartesianRefinedGrid2d grid_coarse( id,id, 0, lx, 0, ly, n, Nx, Ny, bcx, bcy);
    //evaluate on fine grid
    dg::DVec w2dFINE = dg::create::volume( grid_fine);
    dg::DVec v2dFINE = dg::create::inv_weights( grid_fine);
    //evaluate on coarse grid
    dg::DVec w2d = dg::create::volume( grid_coarse);
    dg::DVec v2d = dg::create::inv_weights( grid_coarse);
    //create interpolation and projection matrices
    dg::IDMatrix Q = dg::create::interpolation( grid_fine, grid_coarse);
    dg::IDMatrix P = dg::create::projection( grid_coarse, grid_fine);
    //create functions A(chi) x = b
    dg::DVec x =        dg::evaluate( initial, grid_coarse);
    dg::DVec b =        dg::evaluate( rhs, grid_coarse);
    dg::DVec bFINE =    dg::evaluate( rhs, grid_fine);
    dg::DVec xFINE =    dg::evaluate( rhs, grid_fine);
    dg::blas2::gemv( P, bFINE, b);
    dg::DVec chi     =  dg::evaluate( pol, grid_coarse);
    dg::DVec chiFINE =  dg::evaluate( pol, grid_fine);
    dg::blas2::gemv( Q, chi, chiFINE);
    dg::DVec temp = x;


    std::cout << "Create Polarisation object and set chi!\n";
    t.tic();
    {
        dg::RefinedElliptic<dg::CartesianRefinedGrid2d, dg::IDMatrix, dg::DMatrix, dg::DVec> pol( grid_coarse, grid_fine, dg::not_normed, dg::centered);
        pol.set_chi( chiFINE);
        t.toc();
        std::cout << "Creation of polarisation object took: "<<t.diff()<<"s\n";

        dg::Invert<dg::DVec > invert( x, n*n*Nx*Ny, eps);


        std::cout << eps<<" ";
        t.tic();
        std::cout << " "<< invert( pol, x, b);
        t.toc();
        //std::cout << "Took "<<t.diff()<<"s\n";
    }

    //compute errorFINE
    const dg::DVec solutionFINE = dg::evaluate( sol, grid_fine);
    const dg::DVec derivatiFINE = dg::evaluate( der, grid_fine);
    dg::DVec errorFINE( solutionFINE);
    const dg::DVec solution = dg::evaluate( sol, grid_coarse);
    const dg::DVec derivati = dg::evaluate( der, grid_coarse);
    dg::DVec error( solution);

    dg::blas2::gemv( Q, x, xFINE);
    dg::blas1::axpby( 1.,xFINE,-1., solutionFINE, errorFINE);
    double errFINE = dg::blas2::dot( w2dFINE, errorFINE);
    dg::blas1::axpby( 1.,x,-1., solution, error);
    double err = dg::blas2::dot( w2d, error);
    const double norm = dg::blas2::dot( w2dFINE, solutionFINE);
    std::cout << " "<<sqrt( err/norm) << " " <<sqrt(errFINE/norm);
    {
        dg::RefinedElliptic<dg::CartesianRefinedGrid2d, dg::IDMatrix, dg::DMatrix, dg::DVec> pol_forward( grid_coarse, grid_fine, dg::not_normed, dg::forward);
        pol_forward.set_chi( chiFINE);
        x = temp;
        dg::Invert<dg::DVec > invert_fw( x, n*n*Nx*Ny, eps);
        std::cout << " "<< invert_fw( pol_forward, x, b);
        dg::blas2::gemv( Q, x, xFINE);
        dg::blas1::axpby( 1.,xFINE,-1., solutionFINE, errorFINE);
        errFINE = dg::blas2::dot( w2dFINE, errorFINE);
        dg::blas1::axpby( 1.,x,-1., solution, error);
        err = dg::blas2::dot( w2d, error);
        std::cout << " "<<sqrt( err/norm) << " " <<sqrt(errFINE/norm);
    }

    {
        dg::RefinedElliptic<dg::CartesianRefinedGrid2d, dg::IDMatrix, dg::DMatrix, dg::DVec> pol_backward( grid_coarse, grid_fine, dg::not_normed, dg::backward);
        pol_backward.set_chi( chiFINE);
        x = temp;
        dg::Invert<dg::DVec > invert_bw( x, n*n*Nx*Ny, eps);
        std::cout << " "<< invert_bw( pol_backward, x, b);
        dg::blas2::gemv( Q, x, xFINE);
        dg::blas1::axpby( 1.,xFINE,-1., solutionFINE, errorFINE);
        err = dg::blas2::dot( w2dFINE, errorFINE);
        std::cout << " "<<sqrt( err/norm)<<std::endl;
    }


    dg::DMatrix DX = dg::create::dx( grid_fine);
    dg::blas2::symv( DX, xFINE, errorFINE);
    dg::blas1::axpby( 1.,derivatiFINE,-1., errorFINE);
    err = dg::blas2::dot( w2dFINE, errorFINE);
    //std::cout << "L2 Norm2 of Error in derivatiFINEve is         " << err << std::endl;
    const double norm_der = dg::blas2::dot( w2dFINE, derivatiFINE);
    std::cout << "L2 Norm of relative error in derivative is "<<sqrt( err/norm_der)<<std::endl;
    //derivative converges with p-1, for p = 1 with 1/2

    return 0;
}

