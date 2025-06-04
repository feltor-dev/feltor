#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include "dg/algorithm.h"

#include "tensorelliptic.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;

double initial( double, double) {return 0.;}
double amp = 0.999;
double pol( double x, double y) {return 1. + amp*sin(x)*sin(y); } //must be strictly positive //chi

// double rhs( double x, double y) { return 2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);}   //-div chi nabla phi solution
// double rhs( double x, double y) { return -1.0*(-amp*cos(2.*y) + amp* cos(2.*x) *(-1. + 4. * cos(2.*y)) + 4.*sin(x)*sin(y));}                              //-Div div chi nabla^2 phi solution
// double rhs( double x, double y) { return -1.0*(-2.*amp*cos(2.*y) + 2.* amp* cos(2.*x) *(-1. + 2. * cos(2.*y)) + 4.*sin(x)*sin(y));}                       //-lap chi lap phi solution
// double rhs( double x, double y) { return -1.0*(-2.*amp*cos(y)*cos(y)*sin(x)*sin(x) + sin(y)*(2.*sin(x) + amp* (1.0-3.0*cos(2.*x))*sin(y)));}                       //-Div div chi nabla^2 phi i solution (only diagonal terms)

double rhs( double x, double y) { return 2.0*(-amp*cos(2.*y) + amp* cos(2.*x) *(-1. + 4. * cos(2.*y)) + 4.*sin(x)*sin(y))   -1.0*(-2.*amp*cos(2.*y) + 2.* amp* cos(2.*x) *(-1. + 2. * cos(2.*y)) + 4.*sin(x)*sin(y)) +2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);}                              //full solution - div chi nabla phi - lap chi lap phi + 2 div div chi nabla^2 phi

double sol(double x, double y)  { return sin( x)*sin(y);} //phi


int main()
{
    unsigned n, Nx, Ny;
    double eps;
    double eps_fac;
    double jfactor;

// 	n = 3;
// 	Nx = Ny = 32;
// 	eps = 1e-6;


// 	jfactor = 1;

	std::cout << "# Type n, Nx and Ny, and epsilon, and epsilon_factor (0.1) and jfactor (1)! \n";
    std::cin >> n >> Nx >> Ny; //more N means less iterations for same error
    std::cin >> eps >> eps_fac >> jfactor;
    std::cout <<"# You typed\n"
              <<"n:  "<<n<<"\n"
              <<"Nx: "<<Nx<<"\n"
              <<"Ny: "<<Ny<<"\n"
              <<"eps: "<<eps<<"\n"
              <<"eps_fac: "<<eps_fac<<"\n"
              <<"jfactor: "<<jfactor<<std::endl;
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
    const double norm = dg::blas2::dot( w2d, solution);
    dg::DVec error( solution);

    dg::Timer t;
    const unsigned stages = 3;
    dg::MultigridCG2d<dg::CartesianGrid2d, dg::DMatrix, dg::DVec > multigrid( grid, stages);


    t.tic();
    const std::vector<dg::DVec> multi_chi = multigrid.project( chi);
    std::vector<dg::mat::TensorElliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> > multi_tensorelliptic( stages);
    for(unsigned u=0; u<stages; u++)
    {
        multi_tensorelliptic[u].construct( multigrid.grid(u), dg::centered, jfactor);
        multi_tensorelliptic[u].set_chi( multi_chi[u]);
        multi_tensorelliptic[u].set_iota( multi_chi[u]);
    }
    t.toc();
    std::cout << "# Creation of multigrid took: "<<t.diff()<<"s\n";

    std::cout << "# Inverting tensor elliptic operator\n";
    t.tic();
    std::vector<unsigned> number = multigrid.solve(multi_tensorelliptic, x, b,{eps, eps*eps_fac, eps*eps_fac});
    t.toc();

    std::cout << "time: "<< t.diff() <<"s\n";
    //! [multigrid]
    dg::blas1::axpby( 1.,x,-1., solution, error);
    double err_norm = dg::blas2::dot( w2d, error);
    err_norm = sqrt( err_norm/norm);
    std::cout << "error: "<<err_norm <<"\n";
    std::cout << "iterations[0]: "<<number[0]<<"\n";
    std::cout << "iterations[1]: "<<number[1]<<"\n";
    std::cout << "iterations[2]: "<<number[2]<<"\n";

    return 0;
}

