#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include "backend/timer.h"
#include "topology/projection.h"
#include "dg/file/file.h"

#include "blas.h"
#include "special.h"
#include "elliptic.h"
#include "helmholtz.h"
#include "multigrid.h"

//global relative error in L2 norm is O(h^P)
//as a rule of thumb with n=4 the true error is err = 1e-3 * eps as long as eps > 1e3*err

const double tau = 1.;
const double alpha = -tau/2.;

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;



double initial( double x, double y) {return 0.;}
double amp = 0.999;
double pol( double x, double y) {return 1. + amp*sin(x)*sin(y); } //must be strictly positive //chi

// double rhs( double x, double y) { return 2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);}   //-div chi nabla phi solution 
// double rhs( double x, double y) { return -1.0*(-amp*cos(2.*y) + amp* cos(2.*x) *(-1. + 4. * cos(2.*y)) + 4.*sin(x)*sin(y));}                              //-Div div chi nabla^2 phi solution 
// double rhs( double x, double y) { return -1.0*(-2.*amp*cos(2.*y) + 2.* amp* cos(2.*x) *(-1. + 2. * cos(2.*y)) + 4.*sin(x)*sin(y));}                       //-lap chi lap phi solution 
// double rhs( double x, double y) { return -1.0*(-2.*amp*cos(y)*cos(y)*sin(x)*sin(x) + sin(y)*(2.*sin(x) + amp* (1.0-3.0*cos(2.*x))*sin(y)));}                       //-Div div chi nabla^2 phi i solution (only diagonal terms)

double rhs( double x, double y) { return 2.0*(-amp*cos(2.*y) + amp* cos(2.*x) *(-1. + 4. * cos(2.*y)) + 4.*sin(x)*sin(y))   -1.0*(-2.*amp*cos(2.*y) + 2.* amp* cos(2.*x) *(-1. + 2. * cos(2.*y)) + 4.*sin(x)*sin(y)) +2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);}                              //full solution - div chi nabla phi - lap chi lap phi + 2 div div chi nabla^2 phi 

double rhs2( double x, double y) { return ((amp * cos(2.* y))/(1. - 4.* alpha) + 
 amp* cos(2.* x)* (1./(1. - 4.* alpha) + (10.* cos(2.* y))/(-1. + 8.* alpha)) + (
 12.* sin(x) *sin(y))/(-1. + 2.* alpha ))/(-2. + 4.* alpha);} //with gamma operators

double sol(double x, double y)  { return sin( x)*sin(y);} //phi


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
    const double norm = dg::blas2::dot( w2d, solution);
    dg::DVec error( solution);


    dg::Timer t;



    const unsigned stages = 3;

    dg::MultigridCG2d<dg::CartesianGrid2d, dg::DMatrix, dg::DVec > multigrid( grid, stages);


    t.tic();
    const std::vector<dg::DVec> multi_chi = multigrid.project( chi);

    std::vector<dg::ArbPol<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> > multi_arbpol( stages);
    


    for(unsigned u=0; u<stages; u++)
    {        
        multi_arbpol[u].construct( multigrid.grid(u),  dg::centered, jfactor);
        multi_arbpol[u].set_chi( multi_chi[u]);
        multi_arbpol[u].set_iota( multi_chi[u]);
    }
    t.toc();
    std::cout << "Creation of multigrid took: "<<t.diff()<<"s\n";
    

    std::cout << "#####testing without G1 operators\n";

    t.tic();
    std::vector<unsigned> number = multigrid.direct_solve(multi_arbpol, x, b,{eps, eps*0.1, eps*0.1});
    t.toc();
    std::cout << "Solution took "<< t.diff() <<"s\n";
    for( unsigned u=0; u<number.size(); u++)
    	std::cout << " # iterations stage "<< number.size()-1-u << " " << number[number.size()-1-u] << " \n";
    //! [multigrid]
    dg::blas1::axpby( 1.,x,-1., solution, error);
    double err_norm = dg::blas2::dot( w2d, error);
    err_norm = sqrt( err_norm/norm); 
    std::cout << " "<<err_norm <<"\n";
    
    std::cout << "#####testing with G1 operators\n";
    const dg::DVec b2 =    dg::evaluate( rhs2, grid);
    dg::blas1::scal(x, 0.0);
    dg::Helmholtz< dg::CartesianGrid2d,  dg::DMatrix, dg::DVec  > gamma1inv(grid, alpha, dg::centered, 1.);
    dg::Helmholtz< dg::CartesianGrid2d,  dg::DMatrix, dg::DVec  > gamma1inv_PER(grid, dg::PER, dg::PER, alpha, dg::centered, 1.);

    t.tic();
    dg::blas2::symv(gamma1inv_PER, b2, temp); //b2 fullfills no DIR bc conditions/only PER on [0,2 pi]!
    dg::blas1::pointwiseDot(v2d, temp, temp); //should be normed for multigrid
    number = multigrid.direct_solve(multi_arbpol, x, temp, {eps, eps*0.1, eps*0.1});
    dg::blas2::symv(gamma1inv, x, temp); 
    dg::blas1::pointwiseDot(v2d, temp, x); 
    t.toc();
    std::cout << "Solution with gamma ops took "<< t.diff() <<"s\n";
    for( unsigned u=0; u<number.size(); u++)
    	std::cout << " # iterations stage "<< number.size()-1-u << " " << number[number.size()-1-u] << " \n";
    //! [multigrid]
    dg::blas1::axpby( 1., x, -1., solution, error);
    err_norm = sqrt( dg::blas2::dot( w2d, error)/norm); 
    std::cout << " "<<err_norm <<"\n";
    
    //write into file
//     size_t start = 0;
//     file::NC_Error_Handle err;
//     int ncid;
//     err = nc_create( "visual.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
//     int dim_ids[3], tvarID;
//     err = file::define_dimensions( ncid, dim_ids, &tvarID, grid);
//     
//     std::string names[3] = {"sol", "ana", "error"};
//     int dataIDs[3];
//     for( unsigned i=0; i<3; i++){
//         err = nc_def_var( ncid, names[i].data(), NC_DOUBLE, 3, dim_ids, &dataIDs[i]);}
// 
//     dg::HVec transferH(dg::evaluate(dg::zero, grid));
//     
//     dg::blas1::transfer( x, transferH);
//     file::put_vara_double( ncid, dataIDs[0], start, grid, transferH);
//     dg::blas1::transfer( solution, transferH);
//     file::put_vara_double( ncid, dataIDs[1], start, grid, transferH);
//     dg::blas1::transfer( error, transferH);
//     file::put_vara_double( ncid, dataIDs[2], start, grid, transferH);
//     err = nc_close(ncid);
    return 0;
}

