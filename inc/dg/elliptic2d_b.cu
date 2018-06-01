#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include "backend/timer.cuh"
#include "geometry/projection.cuh"

#include "blas.h"
#include "elliptic.h"
#include "multigrid.h"

//global relative error in L2 norm is O(h^P)
//as a rule of thumb with n=4 the true error is err = 1e-3 * eps as long as eps > 1e3*err

const double lx = M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;

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
    unsigned n, Nx, Ny;
    double eps;
    double jfactor;

	n = 3;
	Nx = Ny = 64;
	eps = 1e-6;
	jfactor = 1;

	/*std::cout << "Type n, Nx and Ny and epsilon and jfactor (1)! \n";
    std::cin >> n >> Nx >> Ny; //more N means less iterations for same error
    std::cin >> eps >> jfactor;*/
    std::cout << "Computation on: "<< n <<" x "<< Nx <<" x "<< Ny << std::endl;
    //std::cout << "# of 2d cells                 "<< Nx*Ny <<std::endl;

	dg::CartesianGrid2d grid( 0, lx, 0, ly, n, Nx, Ny, bcx, bcy);
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
    //compute error
    const dg::DVec solution = dg::evaluate( sol, grid);
    const dg::DVec derivati = dg::evaluate( der, grid);
    const double norm = dg::blas2::dot( w2d, solution);
    dg::DVec error( solution);
    exblas::udouble res;

    //std::cout << "Create Polarisation object and set chi!\n";
    {
    //! [multigrid]
    dg::Timer t;
    t.tic();

    const dg::CartesianGrid2d grid( 0, lx, 0, ly, n, Nx, Ny, bcx, bcy);

    const unsigned stages = 3;

    dg::MultigridCG2d<dg::aGeometry2d, dg::DMatrix, dg::DVec > multigrid( grid, stages);

    const dg::DVec chi =  dg::evaluate( pol, grid);

    const std::vector<dg::DVec> multi_chi = multigrid.project( chi);

    std::vector<dg::Elliptic<dg::aGeometry2d, dg::DMatrix, dg::DVec> > multi_pol( stages);

    for(unsigned u=0; u<stages; u++)
    {
        multi_pol[u].construct( multigrid.grids()[u].get(), dg::not_normed, dg::centered, jfactor);
        multi_pol[u].set_chi( multi_chi[u]);
    }

    t.toc();

    std::cout << "Creation of multigrid took: "<<t.diff()<<"s\n";
    const dg::DVec b =    dg::evaluate( rhs,     grid);
    dg::DVec x       =    dg::evaluate( initial, grid);
    t.tic();
    std::vector<unsigned> number = multigrid.direct_solve(multi_pol, x, b, eps);
    t.toc();
    std::cout << "Solution took "<< t.diff() <<"s\n";
    for( unsigned u=0; u<number.size(); u++)
    	std::cout << " # iterations stage "<< number.size()-1-u << " " << number[number.size()-1-u] << " \n";
    //! [multigrid]
    dg::blas1::axpby( 1.,x,-1., solution, error);
    double err = dg::blas2::dot( w2d, error);
    err = sqrt( err/norm); res.d = err;
    std::cout << " "<<err << "\t"<<res.i<<"\n";
    }


    {
    x = temp;
    //![invert]
    //create an Elliptic object without volume form (not normed)
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> pol_forward( grid, dg::not_normed, dg::centered, jfactor);

    //Set the chi function (chi is a dg::DVec of size grid.size())
    pol_forward.set_chi( chi);

    //construct an invert object
    dg::Invert<dg::DVec > invert_fw( x, n*n*Nx*Ny, eps);

    //invert the elliptic equation
    std::cout << " "<< invert_fw( pol_forward, x, b, w2d, v2d, chi_inv);

    //compute the error (solution contains analytic solution
    dg::blas1::axpby( 1.,x,-1., solution, error);

    //compute the L2 norm of the error
    double err = dg::blas2::dot( w2d, error);

    //output the relative error
    std::cout << " "<<sqrt( err/norm) << "\n";
    //![invert]
    }

    {
		dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> pol_backward( grid, dg::not_normed, dg::backward, jfactor);
		pol_backward.set_chi( chi);
		x = temp;
		dg::Invert<dg::DVec > invert_bw( x, n*n*Nx*Ny, eps);
		std::cout << " "<< invert_bw( pol_backward, x, b, w2d, v2d, chi_inv);
		dg::blas1::axpby( 1.,x,-1., solution, error);
		double err = dg::blas2::dot( w2d, error);
        err = sqrt( err/norm); res.d = err;
        std::cout << " "<<err << "\t"<<res.i<<std::endl;
    }


    dg::DMatrix DX = dg::create::dx( grid);
    dg::blas2::gemv( DX, x, error);
    dg::blas1::axpby( 1.,derivati,-1., error);
    double err = dg::blas2::dot( w2d, error);
    const double norm_der = dg::blas2::dot( w2d, derivati);
    std::cout << "L2 Norm of relative error in derivative is "<<sqrt( err/norm_der)<<std::endl;
    //derivative converges with p-1, for p = 1 with 1/2

    return 0;
}

