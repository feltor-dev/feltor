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

const double lx = M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;

double initial( double, double, double) {return 0.;}
double amp = 0.9;
double pol( double x, double y, double) {return 1. + amp*sin(x)*sin(y); } //must be strictly positive

double rhs( double x, double y, double) { return 2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);}
double sol(double x, double y, double)  { return sin( x)*sin(y);}
double derX(double x, double y, double)  { return cos( x)*sin(y);}
double derY(double x, double y, double)  { return sin( x)*cos(y);}
double vari(double x, double y, double)  { return pol(x,y,0)*pol(x,y,0)*(derX(x,y,0)*derX(x,y,0) + derY(x,y,0)*derY(x,y,0));}


std::function<void(const dg::x::DVec&, dg::x::DVec&)> create_solver(
    dg::x::CartesianGrid3d grid,
    unsigned& number, double eps, unsigned check_every,
    dg::Elliptic<dg::x::CartesianGrid3d,dg::x::DMatrix,dg::x::DVec>& elliptic
    )
{
    dg::x::DVec w3d = dg::create::weights( grid);
    unsigned max_iter = 10000;
    dg::x::DVec precond = elliptic.precond();
    dg::PCG<dg::x::DVec > pcg( w3d, max_iter);
    return [&, pcg, precond, w3d, eps, check_every]( const auto& y, auto& x) mutable
    {
        number = pcg.solve( elliptic, x, y, precond, w3d, eps, 1., check_every);
    };
}

int main(
#ifdef WITH_MPI
    int argc, char* argv[]
#endif
)
{
    unsigned n, Nx, Ny, Nz;
#ifdef WITH_MPI
    dg::mpi_init( &argc, &argv);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    MPI_Comm comm;
    dg::mpi_init3d( bcx, bcy, dg::PER, n, Nx, Ny, Nz, comm);
#else
    std::cout << "# Type n, Nx, Ny and Nz\n";
    std::cin >> n >> Nx >> Ny >> Nz;
#endif
    unsigned num_stages = 5;
    std::vector<double> eps = {1e-6, 0.5*1e-6,0.5*1e-6,0.5*1e-6,0.5*1e-6};
    //unsigned num_stages = 3;
    //std::vector<double> eps = {1e-6, 0.5*1e-6,0.5*1e-6};

    double jfactor = 1;
    DG_RANK0 std::cout <<"# Computation on \n"
              <<"n:  "<<n<<"\n"
              <<"Nx: "<<Nx<<"\n"
              <<"Ny: "<<Ny<<"\n"
              <<"Nz: "<<Nz<<"\n";

	dg::x::CartesianGrid3d grid( 0, lx, 0, ly, 0, 2.*M_PI, n, Nx, Ny, Nz, bcx, bcy, dg::PER
#ifdef WITH_MPI
            , comm
#endif
            );
    dg::x::DVec w3d = dg::create::weights( grid);
    //create functions A(chi) x = b
    dg::x::DVec x =    dg::evaluate( initial, grid);
    dg::x::DVec b =    dg::evaluate( rhs, grid);
    dg::x::DVec chi =  dg::evaluate( pol, grid);
    dg::x::DVec temp = x;
    //compute error
    const dg::x::DVec solution = dg::evaluate( sol, grid);
    const double norm = dg::blas2::dot( w3d, solution);
    dg::x::DVec error( solution);

    dg::NestedGrids<dg::x::CartesianGrid3d, dg::x::DMatrix, dg::x::DVec> nested( grid, num_stages);
    const std::vector<dg::x::DVec> multi_chi = nested.project( chi);
    std::vector<dg::x::DVec> multi_x = nested.project( x);
    std::vector<dg::x::DVec> multi_b = nested.project( b);
    std::vector<dg::Elliptic<dg::x::CartesianGrid3d, dg::x::DMatrix, dg::x::DVec> >
        multi_pol( num_stages);
    std::vector<std::function<void( const dg::x::DVec&, dg::x::DVec&)> >
        multi_inv_pol(num_stages);
    std::vector<unsigned> numbers(num_stages, 0);
    for( unsigned u=0; u<num_stages; u++)
    {
        multi_pol[u].construct( nested.grid(u), dg::forward, jfactor);
        multi_pol[u].set_chi( multi_chi[u]);
        multi_inv_pol[u] = [=, &numbers,
            inverse = create_solver( nested.grid(u),
            numbers[u], eps[u], u == 0 ? 1 : 10, multi_pol[u])] (const auto& y, auto& x) mutable
        {
            dg::Timer t;
            t.tic();
            dg::apply( inverse, y, x);
            t.toc();
            DG_RANK0 std::cout << "-   stage: "<<u<<"\n";
            DG_RANK0 std::cout << "    time: "<<t.diff()<<"\n";
            DG_RANK0 std::cout << "    iter: "<<numbers[u]<<"\n";
        };
    }
    DG_RANK0 std::cout << "stages: \n";
    dg::Timer t;
    t.tic();
    dg::nested_iterations( multi_pol, x, b, multi_inv_pol, nested);
    t.toc();
    DG_RANK0 std::cout <<"time: "<<t.diff()<<"\n";
    DG_RANK0 std::cout <<"iter: "<<numbers[0]<<"\n";
    //compute the error (solution contains analytic solution
    dg::blas1::axpby( 1.,x,-1., solution, error);

    //compute the L2 norm of the error
    double err = dg::blas2::dot( w3d, error);
    DG_RANK0 std::cout << "error: "<<sqrt(err/norm)<<"\n";
    DG_RANK0 std::cout << "error_abs: "<<sqrt(err)<<"\n";


#ifdef WITH_MPI
    MPI_Finalize();
#endif
    return 0;
}

