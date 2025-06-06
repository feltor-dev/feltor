#include <iostream>
#include <iomanip>
#ifdef WITH_MPI
#include <mpi.h>
#include "backend/mpi_init.h"
#endif

#include "arakawa.h"

#include "catch2/catch_test_macros.hpp"


//choose some mean function (attention on lx and ly)
static double m_left( double x, double y) {return sin(x)*exp(x-M_PI)*sin(y);}
static double m_right( double x, double y) {return sin(x)*sin(y)*exp(y-M_PI);}
static double m_jacobian( double x, double y)
{
    return exp( x-M_PI)*(sin(x)+cos(x))*sin(y) * exp(y-M_PI)*sin(x)*(sin(y) +
        cos(y)) - sin(x)*exp(x-M_PI)*cos(y) * cos(x)*sin(y)*exp(y-M_PI);
}

//![function]
static double left( double x, double y) {
    return sin(x)*cos(y);
}
static double right( double x, double y) {
    return sin(y)*cos(x);
}
static double jacobian( double x, double y) {
    return cos(x)*cos(y)*cos(x)*cos(y) - sin(x)*sin(y)*sin(x)*sin(y);
}
//![function]

TEST_CASE( "Arakawa")
{
    const double lx = 2*M_PI;
    const double ly = 2*M_PI;
    const dg::bc bcx = dg::PER;
    const dg::bc bcy = dg::PER;

#ifdef WITH_MPI
    int rank,size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0}, {1,1});
#endif
    INFO("Test the execution of the arakawa scheme!");
    unsigned n = 5, Nx = 32, Ny = 48;
    INFO("Computing on the Grid " <<n<<" x "<<Nx<<" x "<<Ny);

    //![doxygen]
    // create a Cartesian grid on the domain [0,lx]x[0,ly]
    const dg::x::CartesianGrid2d grid( 0, lx, 0, ly, n, Nx, Ny, bcx, bcy
#ifdef WITH_MPI
        , comm
#endif
    );
    // create an Arakawa object
    dg::ArakawaX<dg::x::aGeometry2d, dg::x::DMatrix, dg::x::DVec> arakawa( grid);

    dg::x::DVec lhs, rhs, sol;
    SECTION( "Periodic")
    {
        // evaluate left and right hand side on the grid
        lhs = dg::construct<dg::x::DVec>( dg::evaluate( left, grid));
        rhs = dg::construct<dg::x::DVec>( dg::evaluate( right, grid));
        sol = dg::construct<dg::x::DVec>( dg::evaluate( jacobian, grid));
    }
    SECTION( "Mean")
    {
        lhs = dg::construct<dg::x::DVec>( dg::evaluate( m_left, grid));
        rhs = dg::construct<dg::x::DVec>( dg::evaluate( m_right, grid));
        sol = dg::construct<dg::x::DVec>( dg::evaluate( m_jacobian, grid));
    }

    dg::x::DVec jac(lhs);
    //apply arakawa scheme
    arakawa( lhs, rhs, jac);
    //![doxygen]

    const dg::x::DVec w2d = dg::create::weights( grid);
    //dg::DVec eins = dg::evaluate( dg::one, grid);

    double res = dg::blas2::dot( 1., w2d, jac);
    INFO( "Mean     Jacobian is "<<res);
    CHECK( fabs(res) < 2e-13);
    res = dg::blas2::dot( rhs, w2d, jac);
    INFO( "Mean rhs*Jacobian is "<<res);
    CHECK( fabs(res) < 2e-13);
    res = dg::blas2::dot( lhs, w2d, jac);
    INFO( "Mean lhs*Jacobian is "<<res);
    CHECK( fabs(res) < 2e-13);
    dg::blas1::axpby( 1., sol, -1., jac);
    res = sqrt(dg::blas2::dot( w2d, jac)); //don't forget sqrt when computing errors
    INFO( "Distance to solution "<<res);
    CHECK( res < 2e-3);
    //periocid bc       |  dirichlet bc
    //n = 1 -> p = 2    |
    //n = 2 -> p = 1    |
    //n = 3 -> p = 3    |        3
    //n = 4 -> p = 3    |
    //n = 5 -> p = 5    |
    // quantities are all conserved to 1e-15 for periodic bc
    // for dirichlet bc these are not better conserved than normal jacobian
}
