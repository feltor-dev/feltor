#include <iostream>
#include <iomanip>
#ifdef WITH_MPI
#include <mpi.h>
#include "backend/mpi_init.h"
#endif

#include "poisson.h"

#include "catch2/catch_test_macros.hpp"

//const double lx = 2.*M_PI;
//const double ly = 2.*M_PI;



//choose some mean function (attention on lx and ly)
/*
//THESE ARE NOT PERIODIC
double left( double x, double y) { return sin(x/2)*sin(x/2)*exp(x)*sin(y/2.)*sin(y/2.)*log(y+1); }
double right( double x, double y){ return sin(y/2.)*sin(y/2.)*exp(y)*sin(x/2)*sin(x/2)*log(x+1); }
*/
/*
double left( double x, double y) {return sin(x)*exp(x-M_PI)*sin(y);}
double right( double x, double y) {return sin(x)*sin(y);}
double right2( double x, double y) {return exp(y-M_PI);}
double jacobian( double x, double y)
{
    return exp( x-M_PI)*(sin(x)+cos(x))*sin(y) * exp(y-M_PI)*sin(x)*(sin(y) + cos(y)) - sin(x)*exp(x-M_PI)*cos(y) * cos(x)*sin(y)*exp(y-M_PI);
}
*/

double left( double x, double y) {return sin(x)*cos(y);}
double right( double x, double y) { return sin(y)*cos(x); }
const double lx = M_PI;
const double ly = M_PI;
dg::bc bcxlhs = dg::DIR;
dg::bc bcylhs = dg::NEU;
dg::bc bcxrhs = dg::NEU;
dg::bc bcyrhs = dg::DIR;
//double right2( double x, double y) {return sin(y);}
double jacobian( double x, double y)
{
    return cos(x)*cos(y)*cos(x)*cos(y) - sin(x)*sin(y)*sin(x)*sin(y);
}

TEST_CASE("Poisson")
{
#ifdef WITH_MPI
    int rank,size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0}, {1,1});
    // TODO Does the bcx and bcy in the communicator matter??
#endif
    //![doxygen]
    unsigned n = 5, Nx = 32, Ny = 48;
    INFO("Computing on the Grid " <<n<<" x "<<Nx<<" x "<<Ny);
    const dg::x::CartesianGrid2d grid( 0, lx, 0, ly, n, Nx, Ny
#ifdef WITH_MPI
        , comm
#endif
    );
    const dg::x::DVec lhs = dg::evaluate( left, grid);
    const dg::x::DVec rhs = dg::evaluate( right, grid);
    dg::x::DVec jac(lhs);

    dg::Poisson<dg::x::aGeometry2d, dg::x::DMatrix, dg::x::DVec> poisson( grid,
        bcxlhs, bcylhs, bcxrhs, bcyrhs );
    poisson( lhs, rhs, jac);
    //![doxygen]

    const dg::x::DVec w2d  = dg::create::weights( grid);
    const dg::x::DVec sol  = dg::evaluate ( jacobian, grid);

    double res = dg::blas2::dot( 1., w2d, jac);
    INFO( "Mean     Jacobian is "<<res);
    CHECK( res < 1e-15);
    res = dg::blas2::dot( rhs, w2d, jac);
    INFO( "Mean rhs*Jacobian is "<<res);
    CHECK( res < 1e-15);
    res = dg::blas2::dot( lhs, w2d, jac);
    INFO( "Mean lhs*Jacobian is "<<res);
    CHECK( res < 1e-15);
    dg::blas1::axpby( 1., sol, -1., jac);
    res = sqrt(dg::blas2::dot( w2d, jac)); //don't forget sqrt when computing errors
    INFO( "Distance to solution "<<res);
    CHECK( res < 2e-3);
}
