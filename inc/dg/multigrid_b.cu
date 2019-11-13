#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>

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

//TODO list
//1. Show that the thing reliably converges ( I think we should focus on showing that a single
//FMG sweep reliably produces an acceptable solution even if the discretiation error has not yet been reached)
//2. Can we use this with fixed number of smooting and 1 FMG cycle in simulations?
// (it seems that the number of smoothing steps influences execution time very
// little which means that the CG on the coarse grid dominates time;)
//3. The eps on the lowest grid can be larger than on the fine grids because the discretization error on the coarse grid is h^3 times higher
//4. With forward discretization there seems to be a sweet spot on how many smoothing steps to choose
//5. The relevant errors for us are the gradient in phi errors
//6. The range that the Chebyshev solver smoothes influences the error in the end

int main()
{
    unsigned n = 3, Nx = 32, Ny = 64;
    double eps = 1e-6;
    double jfactor = 1;

    std::cout << "Type n(3) Nx(32) Ny(64)!\n";
    std::cin >> n >> Nx >> Ny;

    std::cout << "Computation on: "<< n <<" x "<< Nx <<" x "<< Ny << std::endl;

    dg::CartesianGrid2d grid( 0, lx, 0, ly, n, Nx, Ny, bcx, bcy);
    dg::DVec w2d = dg::create::weights( grid);
    dg::DVec v2d = dg::create::inv_weights( grid);
    //create functions A(chi) x = b
    dg::DVec x =    dg::evaluate( initial, grid);
    dg::DVec b =    dg::evaluate( rhs, grid);
    const dg::DVec chi =  dg::evaluate( pol, grid);
    const dg::DVec solution = dg::evaluate( sol, grid);

    unsigned stages = 3;
    std::cout<< "Type number of stages (3) and jfactor (10) !\n";
    std::cin >> stages >> jfactor;
    dg::MultigridCG2d<dg::aGeometry2d, dg::DMatrix, dg::DVec > multigrid(
        grid, stages);
    const std::vector<dg::DVec> multi_chi = multigrid.project( chi);

    std::vector<dg::DVec> multi_x = multigrid.project( x);
    const std::vector<dg::DVec> multi_b = multigrid.project( b);
    std::vector<dg::Elliptic<dg::aGeometry2d, dg::DMatrix, dg::DVec> > multi_pol( stages);
    std::vector<dg::EVE<dg::DVec> > multi_eve(stages);
    std::vector<double> multi_ev(stages);
    double eps_ev = 1e-4;
    double hxhy = lx*ly/(n*n*Nx*Ny);
    unsigned counter;
    std::cout << "\nPrecision EVE is "<<eps_ev<<"\n";
    for(unsigned u=0; u<stages; u++)
    {
        multi_pol[u].construct( multigrid.grid(u), dg::not_normed,
            dg::centered, jfactor);
        multi_pol[u].set_chi( multi_chi[u]);
        //estimate EVs
        multi_eve[u].construct( multi_chi[u]);
        counter = multi_eve[u]( multi_pol[u], multi_x[u], multi_b[u],
            multi_ev[u], eps_ev);
        //multi_ev[u]/=hxhy;
        std::cout << "Eigenvalue estimate eve: "<<multi_ev[u]<<"\n";
        std::cout << " with "<<counter<<" iterations\n";
        hxhy*=4;
    }
    std::cout << "\n\n";
    //////////////////////////////setup and write netcdf//////////////////
    err = nc_create( "multigrid.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim3d[3];
    err = file::define_dimensions(  ncid, dim3d, &tvarID, grid, {"step", "y", "x"} );

    err = nc_def_var( ncid, "x_num", NC_DOUBLE, 3, dim3d, &xID);
    err = nc_def_var( ncid, "b_num", NC_DOUBLE, 3, dim3d, &bID);
    err = nc_def_var( ncid, "r_num", NC_DOUBLE, 3, dim3d, &rID);
    ////////////////////////////////////////////////////
    std::cout << "Type nu1 (20), nu2 (20) gamma (1) max_iter (1)\n";
    unsigned nu1, nu2, gamma;
    int max_iter = 3;
    std::cin >> nu1 >> nu2 >> gamma >> max_iter;
    x = dg::evaluate( initial, grid);
    dg::Timer t;
    //for( int i=0; i<max_iter; i++)
    {
        t.tic();
        multigrid.pcg_solve(multi_pol, x, b, multi_ev, nu1, nu2, gamma, eps);
        t.toc();
        std::cout << "Took "<<t.diff()<<"s\n";
        const double norm = dg::blas2::dot( w2d, solution);
        dg::DVec error( solution);
        dg::blas1::axpby( 1.,x,-1., solution, error);
        double err = dg::blas2::dot( w2d, error);
        err = sqrt( err/norm);
        //std::cout << " At iteration "<<i<<"\n";
        std::cout << " Error of Multigrid iterations "<<err<<"\n\n";
    }
    ////////////////////////////////////////////////////

    x = dg::evaluate( initial, grid);
    t.tic();
    multigrid.direct_solve(multi_pol, x, b, eps);
    t.toc();
    const double norm = dg::blas2::dot( w2d, solution);
    dg::DVec error( solution);
    dg::blas1::axpby( 1.,x,-1., solution, error);
    double err = dg::blas2::dot( w2d, error);
    err = sqrt( err/norm);
    std::cout << " Error of nested iterations "<<err<<"\n";
    std::cout << "Took "<<t.diff()<<"s\n\n";
    /////////////////////////////////////////////////////
    int varID;
    err = nc_def_var( ncid, "x_ana", NC_DOUBLE, 2, &dim3d[1], &varID);
    dg::HVec data = solution;
    file::put_var_double( ncid, varID, grid, data);
    err = nc_close( ncid);


    return 0;
}
