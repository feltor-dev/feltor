#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include "backend/timer.h"
#include "topology/projection.h"

#include "blas.h"
#include "special.h"
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
double pol( double x, double y) {return 1. + amp*sin(x)*sin(y); } //must be strictly positive //chi

// double rhs( double x, double y) { return 2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);}   //-div chi nabla phi solution 
// double rhs( double x, double y) { return -1.0*(-amp*cos(2.*y) + amp* cos(2.*x) *(-1. + 4. * cos(2.*y)) + 4.*sin(x)*sin(y));}                              //-Div div chi nabla^2 phi solution 
// double rhs( double x, double y) { return -1.0*(-2.*amp*cos(2.*y) + 2.* amp* cos(2.*x) *(-1. + 2. * cos(2.*y)) + 4.*sin(x)*sin(y));}                       //-lap chi lap phi solution 
// double rhs( double x, double y) { return -1.0*(-2.*amp*cos(y)*cos(y)*sin(x)*sin(x) + sin(y)*(2.*sin(x) + amp* (1.0-3.0*cos(2.*x))*sin(y)));}                       //-Div div chi nabla^2 phi i solution (only diagonal terms)

double rhs( double x, double y) { return 2.0*(-amp*cos(2.*y) + amp* cos(2.*x) *(-1. + 4. * cos(2.*y)) + 4.*sin(x)*sin(y))   -1.0*(-2.*amp*cos(2.*y) + 2.* amp* cos(2.*x) *(-1. + 2. * cos(2.*y)) + 4.*sin(x)*sin(y)) +2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);}                              //full solution - div chi nabla phi - lap chi lap phi + 2 div div chi nabla^2 phi 


double sol(double x, double y)  { return sin( x)*sin(y);} //phi
double der(double x, double y)  { return cos( x)*sin(y);} 


int main()
{
    unsigned n, Nx, Ny;
    double eps;
    double jfactor;

// 	n = 3;
// 	Nx = Ny = 32;
// 	eps = 1e-6;
    
    
// 	jfactor = 1;

	std::cout << "Type n, Nx and Ny and epsilon and jfactor (1)! \n";
    std::cin >> n >> Nx >> Ny; //more N means less iterations for same error
    std::cin >> eps >> jfactor;
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

    std::vector<dg::ArbPol<dg::aGeometry2d, dg::DMatrix, dg::DVec> > multi_arbpol( stages);

    for(unsigned u=0; u<stages; u++)
    {        
        multi_arbpol[u].construct( multigrid.grid(u),  dg::centered, jfactor);
        multi_arbpol[u].set_chi( multi_chi[u]);
        multi_arbpol[u].set_iota( multi_chi[u]);
    }

    t.toc();

    std::cout << "Creation of multigrid took: "<<t.diff()<<"s\n";
    const dg::DVec b =    dg::evaluate( rhs,     grid);
    dg::DVec x       =    dg::evaluate( initial, grid);
    t.tic();
    std::vector<unsigned> number = multigrid.direct_solve(multi_arbpol, x, b,{eps,eps*0.1,eps*0.1});
    t.toc();
    std::cout << "Solution took "<< t.diff() <<"s\n";
    for( unsigned u=0; u<number.size(); u++)
    	std::cout << " # iterations stage "<< number.size()-1-u << " " << number[number.size()-1-u] << " \n";
    //! [multigrid]
    dg::blas1::axpby( 1.,x,-1., solution, error);
    double err = dg::blas2::dot( w2d, error);
    err = sqrt( err/norm); 
    std::cout << " "<<err <<"\n";
    }


    
    
    return 0;
}

