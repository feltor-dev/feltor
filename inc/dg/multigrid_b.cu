#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include "dg/file/nc_utilities.h"
#include "backend/timer.h"

#include "blas.h"
#include "elliptic.h"
#include "multigrid.h"

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

	n = 4;
	Nx = 32, Ny = 64;
	eps = 1e-8;
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
    dg::DVec temp0( x), temp1(x), temp2(x), temp3(x);

    dg::Timer t;
    t.tic();

    //create an Elliptic object without volume form (not normed)
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> pol( grid, dg::not_normed, dg::centered, jfactor);

    //Set the chi function (chi is a dg::DVec of size grid.size())
    pol.set_chi( chi);

    //construct an invert object
    dg::EVE<dg::DVec> eve_cg( x, n*n*Nx*Ny);
    //dg::CG<dg::DVec> eve_cg( x, n*n*Nx*Ny);
    double lmax = M_PI*M_PI*(n*n*Nx*Nx/lx/lx + n*n*Ny*Ny/ly/ly); //Eigenvalues of Laplace
    double hxhy = lx*ly/(n*n*Nx*Ny);
    double chi_max = dg::blas1::reduce( chi, 0, thrust::maximum<double>());
    lmax *= chi_max;
    lmax *= hxhy; //we multiplied the matrix by w2d
    std::cout << "Estimated Eigenvalue Analytical "<<lmax<<"\n";

    //invert the elliptic equation
    double ev_max =0, eps_ev = 1e-4;
    unsigned counter;
    counter = eve_cg( pol, x, b, ev_max, eps_ev);
    std::cout << "\nPrecision is "<<eps_ev<<"\n";
    std::cout << "\nEstimated EigenValue Eve is "<<ev_max<<"\n";
    std::cout << " with "<<counter<<" iterations\n";

    //  Now test multigrid with estimated eigenvalues
    unsigned stages = 3;
    dg::MultigridCG2d<dg::aGeometry2d, dg::DMatrix, dg::DVec > multigrid(
        grid, stages);
    const std::vector<dg::DVec> multi_chi = multigrid.project( chi);
    x = dg::evaluate( initial, grid);
    std::vector<dg::DVec> multi_x = multigrid.project( x);
    const std::vector<dg::DVec> multi_b = multigrid.project( b);
    std::vector<dg::Elliptic<dg::aGeometry2d, dg::DMatrix, dg::DVec> > multi_pol( stages);
    std::vector<dg::EVE<dg::DVec> > multi_eve(stages);
    std::vector<double> multi_ev(stages);
    for(unsigned u=0; u<stages; u++)
    {
        multi_pol[u].construct( multigrid.grid(u), dg::normed, dg::centered, 10*jfactor);
        multi_eve[u].construct( multi_chi[u]);
        multi_pol[u].set_chi( multi_chi[u]);
        counter = multi_eve[u]( multi_pol[u], multi_x[u], multi_b[u],
            multi_ev[u], eps_ev);
        multi_pol[u].construct( multigrid.grid(u), dg::not_normed, dg::centered, 10*jfactor);
        multi_pol[u].set_chi( multi_chi[u]);
        std::cout << "Eigenvalue estimate eve: "<<multi_ev[u]<<"\n";
    }
    //////////////////////////////setup and write netcdf//////////////////
    err = nc_create( "multigrid.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim3d[3];
    err = file::define_dimensions(  ncid, dim3d, &tvarID, grid, {"step", "y", "x"} );

    err = nc_def_var( ncid, "x_num", NC_DOUBLE, 3, dim3d, &xID);
    err = nc_def_var( ncid, "b_num", NC_DOUBLE, 3, dim3d, &bID);
    err = nc_def_var( ncid, "r_num", NC_DOUBLE, 3, dim3d, &rID);
    ////////////////////////////////////////////////////
    std::cout << "Type nu1 (3), nu2 (3) gamma (1) max_iter (3)\n";
    unsigned nu1, nu2, gamma;
    int max_iter = 3;
    std::cin >> nu1 >> nu2 >> gamma >> max_iter;
    x = dg::evaluate( initial, grid);
    const dg::DVec solution = dg::evaluate( sol, grid);
    for( int i=0; i<max_iter; i++)
    {
        multigrid.solve(multi_pol, x, b, multi_ev, nu1, nu2, gamma, eps);
        //CURRENTLY BEST METHOD:
        //multigrid.direct_solve(multi_pol, x, b, eps);
        const double norm = dg::blas2::dot( w2d, solution);
        dg::DVec error( solution);
        dg::blas1::axpby( 1.,x,-1., solution, error);
        double err = dg::blas2::dot( w2d, error);
        err = sqrt( err/norm);
        std::cout << " At iteration "<<i<<"\n";
        std::cout << " Error of Multigrid iterations "<<err<<"\n";
    }
    x = dg::evaluate( initial, grid);
    multigrid.direct_solve(multi_pol, x, b, eps);
    const double norm = dg::blas2::dot( w2d, solution);
    dg::DVec error( solution);
    dg::blas1::axpby( 1.,x,-1., solution, error);
    double err = dg::blas2::dot( w2d, error);
    err = sqrt( err/norm);
    std::cout << " Error of nested iterations "<<err<<"\n";

    int varID;
    err = nc_def_var( ncid, "x_ana", NC_DOUBLE, 2, &dim3d[1], &varID);
    dg::HVec data = solution;
    file::put_var_double( ncid, varID, grid, data);
    err = nc_close( ncid);


    return 0;
}
