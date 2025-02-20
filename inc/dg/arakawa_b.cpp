#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#ifdef WITH_MPI
#include <mpi.h>
#include "backend/mpi_init.h"
#include "topology/mpi_evaluation.h"
#include "topology/mpi_weights.h"
#endif

#include "backend/timer.h"
#include "arakawa.h"
#include "blas.h"



const double lx = 2*M_PI;
const double ly = 2*M_PI;
//const double lx = 1.;
//const double ly = 1.;


//choose some mean function (attention on lx and ly)
//THESE ARE NOT PERIODIC
/*
double left( double x, double y) { return sin(x)*cos(y);}
double right( double x, double y){ return exp(0.1*(x+y)); }
double jacobian( double x, double y)
{
    return exp( x-M_PI)*(sin(x)+cos(x))*sin(y) * exp(y-M_PI)*sin(x)*(sin(y) + cos(y)) - sin(x)*exp(x-M_PI)*cos(y) * cos(x)*sin(y)*exp(y-M_PI);
}
*/

dg::bc bcx = dg::PER;
dg::bc bcy = dg::PER;
double left( double x, double y) {return sin(x)*cos(y);}
double right( double x, double y) {return cos(x)*sin(y);}
double jacobian( double x, double y)
{
    return cos(x)*cos(y)*cos(x)*cos(y) - sin(x)*sin(y)*sin(x)*sin(y);
}
////These are for comparing to FD arakawa results
//double left( double x, double y) {return sin(2.*M_PI*(x-hx/2.));}
//double right( double x, double y) {return y;}
//double jacobian( double x, double y) {return 2.*M_PI*cos(2.*M_PI*(x-hx/2.));}

// 3d functions
const double lz = 1.;
double left( double x, double y, double z) {return left(x,y)*z;}
double right( double x, double y, double z) {return right(x,y)*z;}
//double right2( double x, double y) {return sin(y);}
double jacobian( double x, double y, double z)
{
    return jacobian(x,y)*z*z;
}

//using Vector = std::array<dg::DVec, 10>;
using Vector = dg::x::DVec;
using Matrix = dg::x::DMatrix;

int main(int argc, char* argv[])
{
    unsigned n, Nx, Ny;
#ifdef WITH_MPI
    MPI_Init( &argc, &argv);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
    { // 2d tests
    DG_RANK0 std::cout << std::fixed<<"#Benchmark 2d Arakawa!\n";
#ifdef WITH_MPI
    MPI_Comm comm;
    dg::mpi_init2d( bcx, bcy, n, Nx, Ny, comm);
#else
    std::cout << "Type n, Nx and Ny! \n";
    std::cin >> n >> Nx >> Ny;
#endif
    DG_RANK0 std::cout << "#Computing on the Grid " <<n<<" x "<<Nx<<" x "<<Ny <<std::endl;
    dg::Timer t;
    dg::x::Grid2d grid( 0, lx, 0, ly, n, Nx, Ny, dg::PER, dg::PER
#ifdef WITH_MPI
    , comm
#endif
    );
    Vector w2d = dg::create::weights( grid);
    Vector lhs = dg::construct<Vector>(dg::evaluate ( left, grid)), jac(lhs);
    Vector rhs = dg::construct<Vector>(dg::evaluate ( right,grid));
    const Vector sol = dg::construct<Vector>(dg::evaluate( jacobian, grid ));
    Vector eins = dg::construct<Vector>(dg::evaluate( dg::one, grid ));
    //std::cout<< std::setprecision(2);

    dg::ArakawaX<dg::x::CartesianGrid2d, Matrix, Vector> arakawa( grid);
    unsigned multi=100;
    t.tic();
    for( unsigned i=0; i<multi; i++)
        arakawa( lhs, rhs, jac);
    t.toc();
    DG_RANK0 std::cout << "Arakawa 2d took "<<t.diff()*1000/(double)(multi)<<"ms\n";

    DG_RANK0 std::cout << std::scientific;
    double result = dg::blas2::dot( eins, w2d, jac);
    DG_RANK0 std::cout << "Mean     Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( rhs,  w2d, jac);
    DG_RANK0 std::cout << "Mean rhs*Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( lhs,  w2d, jac);
    DG_RANK0 std::cout << "Mean lhs*Jacobian is "<<result<<"\n";
    dg::blas1::axpby( 1., sol, -1., jac);
    result = sqrt( dg::blas2::dot( w2d, jac));
    DG_RANK0 std::cout << "Distance to solution "<<result<<std::endl;

    //periocid bc       |  dirichlet in x per in y
    //n = 1 -> p = 2    |        1.5
    //n = 2 -> p = 1    |        1
    //n = 3 -> p = 3    |        3
    //n = 4 -> p = 3    |        3
    //n = 5 -> p = 5    |        5
    // quantities are all conserved to 1e-15 for periodic bc
    // for dirichlet bc these are not better conserved than normal jacobian
    } // 2d tests
    { // 3d tests
    DG_RANK0 std::cout << std::fixed<<"#Benchmark 3d Arakawa!\n";
#ifdef WITH_MPI
    unsigned n, Nx, Ny, Nz;
    MPI_Comm comm;
    mpi_init3d( bcx, bcy,dg::PER, n, Nx, Ny,Nz, comm);
#else
    unsigned n, Nx, Ny, Nz;
    std::cout << "Type n, Nx, Ny and Nz! \n";
    std::cin >> n >> Nx >> Ny >> Nz;
#endif
    DG_RANK0 std::cout << "#Computing on the Grid " <<n<<" x "<<Nx<<" x "<<Ny
                       <<" x "<<Nz<<std::endl;
    dg::x::Grid3d grid( 0, lx, 0, ly,0,lz, n, Nx, Ny,Nz, dg::PER, dg::PER, dg::PER
#ifdef WITH_MPI
    , comm
#endif
    );
    Vector w3d = dg::create::weights( grid);
    Vector lhs = dg::evaluate ( left, grid), jac(lhs);
    Vector rhs = dg::evaluate ( right,grid);
    const Vector sol = dg::evaluate( jacobian, grid );
    Vector eins = dg::evaluate( dg::one, grid);
    dg::Timer t;

    dg::ArakawaX<dg::x::CartesianGrid3d, Matrix, Vector> arakawa( grid);
    t.tic();
    for( unsigned i=0; i<20; i++)
        arakawa( lhs, rhs, jac);
    t.toc();
    DG_RANK0 std::cout << "\nArakawa took "<<t.diff()/0.02<<"ms\n";
    DG_RANK0 std::cout <<   "which is     "<<t.diff()/0.02/Nz<<"ms per z plane \n\n";

    DG_RANK0 std::cout << std::scientific;
    double result = dg::blas2::dot( eins, w3d, jac);
    std::cout << std::scientific;
    DG_RANK0 std::cout << "Mean     Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( rhs,  w3d, jac);
    DG_RANK0 std::cout << "Mean rhs*Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( lhs,  w3d, jac);
    DG_RANK0 std::cout << "Mean lhs*Jacobian is "<<result<<"\n";
    dg::blas1::axpby( 1., sol, -1., jac);
    result = sqrt( dg::blas2::dot( w3d, jac));
    DG_RANK0 std::cout << "Distance to solution "<<result<<std::endl;
    } // 3d tests
#ifdef WITH_MPI
    MPI_Finalize();
#endif

    return 0;
}
