#include <iostream>
#include <iomanip>

//#include "backend/xspacelib.cuh"
#include <thrust/device_vector.h>
#include "blas.h"


#include "elliptic.h"
#include "cg.h"
#include "backend/timer.cuh"
#include "backend/projection.cuh"

//NOTE: IF DEVICE=CPU THEN THE POLARISATION ASSEMBLY IS NOT PARALLEL AS IT IS NOW 

//global relative error in L2 norm is O(h^P)
//as a rule of thumb with n=4 the true error is err = 1e-3 * eps as long as eps > 1e3*err

const double lx = M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;
//const double eps = 1e-3; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

double initial( double x, double y) {return 0.;}
double amp = 0.9999;
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
    double eps;
    double jfactor;
    std::cout << "Type n, Nx and Ny and epsilon and jfactor (1)! \n";
    std::cin >> n >> Nx >> Ny; //more N means less iterations for same error
    std::cin >> eps >> jfactor;
    std::cout << "Computation on: "<< n <<" x "<<Nx<<" x "<<Ny<<std::endl;
    //std::cout << "# of 2d cells                 "<< Nx*Ny <<std::endl;
    dg::Grid2d grid( 0, lx, 0, ly, n, Nx, Ny, bcx, bcy);
    dg::DVec w2d = dg::create::weights( grid);
    dg::DVec v2d = dg::create::inv_weights( grid);
    dg::DVec one = dg::evaluate( dg::one, grid);
    //create functions A(chi) x = b
    dg::DVec x =    dg::evaluate( initial, grid);
    dg::DVec b =    dg::evaluate( rhs, grid);
    dg::DVec chi =  dg::evaluate( pol, grid);
    dg::DVec chi_inv(chi); 
    dg::blas1::transform( chi, chi_inv, dg::INVERT<double>());
    dg::blas1::pointwiseDot( chi_inv, v2d, chi_inv);
    dg::DVec temp = x;


    std::cout << "Create Polarisation object and set chi!\n";
    t.tic();
    {
    std::vector<dg::Grid2d> g( 4, grid);
    g[1].multiplyCellNumbers( 0.5, 0.5);
    g[2].multiplyCellNumbers( 0.25, 0.25);
    g[3].multiplyCellNumbers( 0.125, 0.125);
    dg::IDMatrix project3 = dg::create::projection( g[3], g[0]);
    dg::IDMatrix project2 = dg::create::projection( g[2], g[0]);
    dg::IDMatrix project1 = dg::create::projection( g[1], g[0]);
    dg::IDMatrix inter01 = dg::create::interpolation( g[0], g[1]);
    dg::IDMatrix inter12 = dg::create::interpolation( g[1], g[2]);
    dg::IDMatrix inter23 = dg::create::interpolation( g[2], g[3]);
    std::vector<dg::DVec> w2d_(4, w2d), v2d_(4,v2d);
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec>  pol0(g[0], dg::not_normed,dg::centered, jfactor);
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec>  pol1(g[1], dg::not_normed,dg::centered, jfactor);
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec>  pol2(g[2], dg::not_normed,dg::centered, jfactor);
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec>  pol3(g[3], dg::not_normed,dg::centered, jfactor);
    for( unsigned i=0; i<4; i++)
    {
        w2d_[i] = dg::create::weights( g[i]);
        v2d_[i] = dg::create::inv_weights( g[i]);
    }
    std::vector<dg::DVec> chi_(w2d_), chi_inv_(chi_), b_(chi_), x_(chi_);
    dg::blas2::symv( project3, chi, chi_[3]);
    dg::blas2::symv( project2, chi, chi_[2]);
    dg::blas2::symv( project1, chi, chi_[1]);
    dg::blas2::symv( project3, chi_inv, chi_inv_[3]);
    dg::blas2::symv( project2, chi_inv, chi_inv_[2]);
    dg::blas2::symv( project1, chi_inv, chi_inv_[1]);
    dg::blas2::symv( project3, b, b_[3]);
    dg::blas2::symv( project2, b, b_[2]);
    dg::blas2::symv( project1, b, b_[1]);
    dg::blas2::symv( project3, x, x_[3]);
    dg::blas2::symv( project2, x, x_[2]);
    dg::blas2::symv( project1, x, x_[1]);
    pol0.set_chi( chi);
    pol1.set_chi( chi_[1]);
    pol2.set_chi( chi_[2]);
    pol3.set_chi( chi_[3]);
    t.toc();
    std::cout << "Creation of polarisation object took: "<<t.diff()<<"s\n";

    std::vector<dg::CG<dg::DVec > > cg(4);
    cg[0].construct( x, g[0].size());
    for( unsigned i=1; i<4; i++)
        cg[i].construct(x_[i], g[i].size());
    std::cout << eps<<" \n";
    t.tic();
    //dg::blas2::symv( w2d_[3], b_[3], b_[3]); 
    //std::cout << " # iterations grid "<< cg[3]( pol3, x_[3], b_[3], chi_inv_[3], v2d_[3], eps/10) << " \n";
    //dg::blas2::symv( inter23, x_[3], x_[2]);
    dg::blas2::symv( w2d_[2], b_[2], b_[2]); 
    std::cout << " # iterations grid "<< cg[2]( pol2, x_[2], b_[2], chi_inv_[2], v2d_[2], eps/10) << " \n";
    dg::blas2::symv( inter12, x_[2], x_[1]);
    dg::blas2::symv( w2d_[1], b_[1], b_[1]); 
    std::cout << " # iterations grid "<< cg[1]( pol1, x_[1], b_[1], chi_inv_[1], v2d_[1], eps/10) << " \n";
    dg::blas2::symv( inter01, x_[1], x);
    dg::blas2::symv( w2d, b, b_[0]); 
    std::cout << " # iterations fine grid "<< cg[0]( pol0, x, b_[0], chi_inv, v2d, eps)<<std::endl;
    t.toc();
    //std::cout << "Took "<<t.diff()<<"s\n";
    }

    //compute error
    const dg::DVec solution = dg::evaluate( sol, grid);
    const dg::DVec derivati = dg::evaluate( der, grid);
    dg::DVec error( solution);

    dg::blas1::axpby( 1.,x,-1., solution, error);
    double err = dg::blas2::dot( w2d, error);
    //std::cout << "L2 Norm2 of Error is                       " << err << std::endl;
    const double norm = dg::blas2::dot( w2d, solution);
    std::cout << " "<<sqrt( err/norm);
    {
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> pol_forward( grid, dg::not_normed, dg::forward, jfactor);
    pol_forward.set_chi( chi);
    x = temp;
    dg::Invert<dg::DVec > invert_fw( x, n*n*Nx*Ny, eps);
    std::cout << " "<< invert_fw( pol_forward, x, b, v2d, chi_inv);
    dg::blas1::axpby( 1.,x,-1., solution, error);
    err = dg::blas2::dot( w2d, error);
    std::cout << " "<<sqrt( err/norm);
    }

    {
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> pol_backward( grid, dg::not_normed, dg::backward, jfactor);
    pol_backward.set_chi( chi);
    x = temp;
    dg::Invert<dg::DVec > invert_bw( x, n*n*Nx*Ny, eps);
    std::cout << " "<< invert_bw( pol_backward, x, b, v2d, chi_inv);
    dg::blas1::axpby( 1.,x,-1., solution, error);
    err = dg::blas2::dot( w2d, error);
    std::cout << " "<<sqrt( err/norm)<<std::endl;
    }


    dg::DMatrix DX = dg::create::dx( grid);
    dg::blas2::gemv( DX, x, error);
    dg::blas1::axpby( 1.,derivati,-1., error);
    err = dg::blas2::dot( w2d, error);
    //std::cout << "L2 Norm2 of Error in derivative is         " << err << std::endl;
    const double norm_der = dg::blas2::dot( w2d, derivati);
    std::cout << "L2 Norm of relative error in derivative is "<<sqrt( err/norm_der)<<std::endl;
    //derivative converges with p-1, for p = 1 with 1/2

    return 0;
}

