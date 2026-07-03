#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#ifdef WITH_MPI
#include <mpi.h>
#include "backend/mpi_init.h"
#endif

#include "backend/timer.h"
#include "topology/projection.h"

#include "blas.h"
#include "elliptic.h"
#include "multigrid.h"
#include "lgmres.h"
#include "bicgstabl.h"
#include "andersonacc.h"

//global relative error in L2 norm is O(h^P)
//as a rule of thumb with n=4 the true error is err = 1e-3 * eps as long as eps > 1e3*err

const double lx = M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;

double initial( double) {return 0.;}
double initial( double, double) {return 0.;}
//double amp = 0.9999; // LGMRES has problems here
double amp = 0.9;
double pol( double x) {return 1. + amp*sin(x); } //must be strictly positive
double pol( double x, double y) {return 1. + amp*sin(x)*sin(y); } //must be strictly positive
//double pol( double x, double y) {return 1.; }
//double pol( double x, double y) {return 1. + sin(x)*sin(y) + x; } //must be strictly positive

double rhs( double x) { return sin(x) + amp*sin(x)*sin(x) - amp*cos(x)*cos(x);}
double rhs( double x, double y) { return 2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);}
//double rhs( double x, double y) { return 2.*sin( x)*sin(y);}
//double rhs( double x, double y) { return 2.*sin(x)*sin(y)*(sin(x)*sin(y)+1)-sin(x)*sin(x)*cos(y)*cos(y)-cos(x)*cos(x)*sin(y)*sin(y)+(x*sin(x)-cos(x))*sin(y) + x*sin(x)*sin(y);}
double sol(double x)  { return sin( x);}
double sol(double x, double y)  { return sin( x)*sin(y);}
double derX(double x, double y)  { return cos( x)*sin(y);}
double derY(double x, double y)  { return sin( x)*cos(y);}
double vari(double x, double y)  { return pol(x,y)*pol(x,y)*(derX(x,y)*derX(x,y) + derY(x,y)*derY(x,y));}


int main(
#ifdef WITH_MPI
    int argc, char* argv[]
#endif
)
{
#ifdef WITH_MPI
    dg::mpi_init( &argc, &argv);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm comm1d = dg::mpi_cart_create( MPI_COMM_WORLD, {0}, {1});
    MPI_Comm comm2d = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0}, {1,1});
    MPI_Comm comm3d = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0,0}, {1,1,1});
#endif
    unsigned n, Nx, Ny;
    double eps;
    double jfactor;

	n = 3;
	Nx = Ny = 64;
	eps = 1e-6;
	jfactor = 1;

    //std::cout << "Type n, Nx and Ny and epsilon and jfactor (1)! \n";
    //std::cin >> n >> Nx >> Ny; //more N means less iterations for same error
    //std::cin >> eps >> jfactor;
    bool jump_weight = false;
    //std::cout << "Jump weighting on or off? Type 1 for true or 0 for false (default): \n";
    //std::cin >> jump_weight;
    DG_RANK0 std::cout << "Computation on: "<< n <<" x "<< Nx <<" x "<< Ny 
                       << std::endl;
    //std::cout << "# of 2d cells                 "<< Nx*Ny <<std::endl;

	dg::x::CartesianGrid2d grid( 0, lx, 0, ly, n, Nx, Ny, bcx, bcy
#ifdef WITH_MPI
            , comm2d
#endif
            );
    dg::x::DVec w2d = dg::create::weights( grid);
    //create functions A(chi) x = b
    dg::x::DVec x =    dg::evaluate( initial, grid);
    dg::x::DVec b =    dg::evaluate( rhs, grid);
    dg::x::DVec chi =  dg::evaluate( pol, grid);
    dg::x::DVec chi_inv(chi);
    dg::blas1::transform( chi, chi_inv, dg::INVERT<double>());
    dg::x::DVec temp = x;
    //compute error
    const dg::x::DVec solution = dg::evaluate( sol, grid);
    const dg::x::DVec derivati = dg::evaluate( derX, grid);
    const dg::x::DVec variatio = dg::evaluate( vari, grid);
    const double norm = dg::blas2::dot( w2d, solution);
    dg::x::DVec error( solution);
    dg::exblas::udouble res;

    DG_RANK0 std::cout << "Create Polarisation object and set chi!\n";
    {
    DG_RANK0 std::cout << "Centered Elliptic Multigrid\n";
    //! [multigrid]
    dg::Timer t;
    t.tic();

    const dg::x::CartesianGrid2d grid( 0, lx, 0, ly, n, Nx, Ny, bcx, bcy
#ifdef WITH_MPI
            , comm2d
#endif
            );

    const unsigned stages = 3;

    dg::MultigridCG2d<dg::x::aGeometry2d, dg::x::DMatrix, dg::x::DVec >
        multigrid( grid, stages);

    const dg::x::DVec chi =  dg::evaluate( pol, grid);

    const std::vector<dg::x::DVec> multi_chi = multigrid.project( chi);

    std::vector<dg::Elliptic<dg::x::aGeometry2d, dg::x::DMatrix, dg::x::DVec> >
        multi_pol( stages);

    for(unsigned u=0; u<stages; u++)
    {
        multi_pol[u].construct( multigrid.grid(u), dg::centered, jfactor);
        //this tests if elliptic can recover from NaN in the preconditioner
        multi_pol[u].set_chi(0.);
        // here we test if we can set the tensor part in elliptic
        multi_pol[u].set_chi( multigrid.grid(u).metric());
        // now set the actual scalar part in chi
        multi_pol[u].set_chi( multi_chi[u]);
        multi_pol[u].set_jump_weighting(jump_weight);
    }

    t.toc();

    DG_RANK0 std::cout << "Creation of multigrid took: "<<t.diff()<<"s\n";
    const dg::x::DVec b =    dg::evaluate( rhs,     grid);
    dg::x::DVec x       =    dg::evaluate( initial, grid);
    t.tic();
    std::vector<unsigned> number = multigrid.solve(multi_pol, x, b, {eps, 1.5*eps, 1.5*eps});
    t.toc();
    DG_RANK0 std::cout << "Solution took "<< t.diff() <<"s\n";
    for( unsigned u=0; u<number.size(); u++)
    	DG_RANK0 std::cout << " # iterations stage "<< number.size()-1-u << " " << number[number.size()-1-u] << " \n";
    //! [multigrid]
    dg::blas1::axpby( 1.,x,-1., solution, error);
    double err = dg::blas2::dot( w2d, error);
    err = sqrt( err/norm); res.d = err;
    DG_RANK0 std::cout << " "<<err << "\t"<<res.i<<"\n";
    dg::x::DMatrix DX = dg::create::dx( grid);
    dg::blas2::gemv( DX, x, error);
    dg::blas1::axpby( 1.,derivati,-1., error);
    err = dg::blas2::dot( w2d, error);
    const double norm_der = dg::blas2::dot( w2d, derivati);
    DG_RANK0 std::cout << "L2 Norm of relative error in derivative is\n"
        <<std::setprecision(16)<< sqrt( err/norm_der)<<std::endl;
    //derivative converges with p-1, for p = 1 with 1/2

    }
    {
    DG_RANK0 std::cout << "Forward Elliptic\n";
    x = temp;
    //![pcg]
    //create an Elliptic object
    dg::Elliptic<dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec>
        pol_forward( grid, dg::forward, jfactor);

    //Set the chi function (chi is a dg::x::DVec of size grid.size())
    pol_forward.set_chi( chi);

    //construct an pcg object
    dg::PCG pcg( x, n*n*Nx*Ny);

    //invert the elliptic equation
    pcg.solve( pol_forward, x, b, chi_inv, w2d, eps);

    //compute the error (solution contains analytic solution
    dg::blas1::axpby( 1.,x,-1., solution, error);

    //compute the L2 norm of the error
    double err = dg::blas2::dot( w2d, error);

    //output the relative error
    DG_RANK0 std::cout << " "<<sqrt( err/norm) << "\n";
    //![pcg]
    DG_RANK0 std::cout << "Compute variation in forward Elliptic\n";
    pol_forward.variation( 1., chi, x, 0., error);
    dg::blas1::axpby( 1., variatio, -1., error);
    err = dg::blas2::dot( w2d, error);
    const double norm_var = dg::blas2::dot( w2d, variatio);
    DG_RANK0 std::cout << " "<<sqrt( err/norm_var) << "\n";
    DG_RANK0 std::cout << "Compute direct application of forward Elliptic (supraconvergence)\n";
    dg::apply( pol_forward, solution, x);
    dg::blas1::axpby( 1.,x,-1., b, error);
    err = sqrt( dg::blas2::dot(w2d, error));
    DG_RANK0 std::cout << " "<< err << "\n";
    // NOW TEST LGMRES AND BICGSTABl
    unsigned inner_m = 30, outer_k = 3;
    //std::cout << " Type inner and outer iterations (30 3)!\n";
    //std::cin >> inner_m >> outer_k;
    dg::LGMRES lgmres( x, inner_m, outer_k, 10000/inner_m);
    dg::blas1::copy( 0., x);
    dg::Timer t;
    t.tic();
    //unsigned number = lgmres.solve( pol_forward, x, b, chi_inv, w2d, eps);
    unsigned number = lgmres.solve( pol_forward, x, b, 1., w2d, eps);
    t.toc();
    DG_RANK0 std::cout << "# of lgmres iterations "<<number<<" took "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1.,x,-1., solution, error);
    err = dg::blas2::dot( w2d, error);
    DG_RANK0 std::cout << " "<<sqrt( err/norm) << "\n";
    unsigned l_input = 2;
    dg::BICGSTABl bicg( x, 100000, l_input);
    dg::blas1::copy( 0., x);
    t.tic();
    number = bicg.solve( pol_forward, x, b, chi_inv, w2d, eps);
    //number = bicg.solve( pol_forward, x, b, 1., w2d, eps);
    t.toc();
    DG_RANK0 std::cout << "# of bicgstabl iterations "<<number<<" took "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1.,x,-1., solution, error);
    err = dg::blas2::dot( w2d, error);
    DG_RANK0 std::cout << " "<<sqrt( err/norm) << "\n";
    unsigned mMax = 8;
    double damping = 1e-5;
    double restart = 8;
    //std::cout << "Type mMAx (8), damping ( 1e-5), restart (8)\n";
    //std::cin >> mMax >> damping >> restart;
    dg::AndersonAcceleration anderson( x, mMax);
    dg::blas1::copy( 0., x);
    t.tic();
    number = anderson.solve( pol_forward, x, b, w2d, eps, eps, 3000, damping, restart, false );
    t.toc();
    DG_RANK0 std::cout << "# of anderson iterations "<<number<<" took "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1.,x,-1., solution, error);
    err = dg::blas2::dot( w2d, error);
    DG_RANK0 std::cout << " "<<sqrt( err/norm) << "\n";
    }

    dg::PCG<dg::x::DVec > pcg( x, n*n*Nx*Ny);

    {
        DG_RANK0 std::cout << "Compute 2d handle of Elliptic3d\n";
        dg::x::CartesianGrid3d grid( 0, lx, 0, ly, 0,1,n, Nx, Ny, 1, bcx, bcy,
                dg::PER
#ifdef WITH_MPI
                , comm3d
#endif
                );
        dg::Elliptic3d<dg::x::CartesianGrid3d, dg::x::DMatrix, dg::x::DVec>
            pol_backward( grid, dg::backward, jfactor);
        pol_backward.set_compute_in_2d(true);
		pol_backward.set_chi( chi);
		x = temp;
		pcg.solve( pol_backward, x, b, chi_inv, w2d, eps);
		dg::blas1::axpby( 1.,x,-1., solution, error);
		double err = dg::blas2::dot( w2d, error);
        err = sqrt( err/norm); res.d = err;
        DG_RANK0 std::cout << " "<<err << "\t"<<res.i<<std::endl;
    }
    {
        //try the Hyperelliptic operator
        DG_RANK0 std::cout << "HyperElliptic operator\n";
	    dg::x::CartesianGrid2d grid( 0, lx, 0, ly,n, Nx, Ny, bcx, bcy
#ifdef WITH_MPI
                , comm2d
#endif
);
        dg::Elliptic<dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec>
            pol_backward( grid, dg::backward, jfactor);
		x = temp;
        chi = temp;
		x = temp;
		pcg.solve( pol_backward, chi, solution, 1., w2d, eps);
		pcg.solve( pol_backward, x, chi, 1., w2d, eps);
		dg::blas1::axpby( 1.,x,-0.25, solution, error);
		double err = dg::blas2::dot( w2d, error);
        err = sqrt( err/norm); res.d = err;
        DG_RANK0 std::cout << " "<<err << "\t"<<res.i<<std::endl;
    }
    try{
        dg::blas1::copy( 0., x);
        dg::Elliptic<dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec>
            pol_f( grid);
        // We expect it not to converge
        dg::LGMRES lgmres( x, 10, 3, 1);
        lgmres.solve( pol_f, x, b, 1., w2d, eps);
    }catch( dg::Fail& fail)
    {
        DG_RANK0 std::cerr << "TEST failure message";
        DG_RANK0 std::cerr << fail.what()<<std::endl;
        DG_RANK0 std::cerr << "END TEST failure message\n";
    }
    {
        DG_RANK0 std::cout  <<"# TEST 1D elliptic operator\n";
        dg::x::Grid1d grid1d( 0, lx, n, Nx, bcx
#ifdef WITH_MPI
                , comm1d
#endif
                );
        dg::x::DVec w1d = dg::create::weights( grid1d);
        x =    dg::evaluate( initial, grid1d);
        b =    dg::evaluate( rhs, grid1d);
        chi =  dg::evaluate( pol, grid1d);
        const dg::x::DVec sol1d = dg::evaluate( sol, grid1d);
        const double norm1d = dg::blas2::dot( w1d, sol1d);
        chi_inv = chi;
        dg::blas1::transform( chi, chi_inv, dg::INVERT<double>());
        dg::x::DVec temp = x;
        dg::Elliptic1d<dg::x::Grid1d, dg::x::DMatrix, dg::x::DVec> pol1d(
                grid1d, dg::backward, jfactor);
        pol1d.set_chi( chi);
        dg::PCG<dg::x::DVec > pcg1d( x, 10*n*Nx);
        unsigned number = 0;
        try{
		    number = pcg1d.solve( pol1d, x, b, chi_inv, w1d, eps);
        }catch ( dg::Fail& fail){}
        DG_RANK0 std::cout << "# Number of 1d iterations: "<<number<<"\n";

        dg::blas1::axpby( 1.,x,-1., sol1d, x);
        double err = dg::blas2::dot( w1d, x);
        err = sqrt( err/norm1d); res.d = err;
        DG_RANK0 std::cout << " "<<err<<"\n";
        DG_RANK0 std::cout << "Compute direct application of 1d Elliptic (supraconvergence)\n";
        dg::apply( pol1d, sol1d, x);
        dg::blas1::axpby( 1.,x,-1., b, x);
        err = sqrt( dg::blas2::dot( w1d, x));
        DG_RANK0 std::cout << " "<< err << "\n";

    }



#ifdef WITH_MPI
    MPI_Finalize();
#endif
    return 0;
}

