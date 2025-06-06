#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#ifdef WITH_MPI
#include <mpi.h>
#include "backend/mpi_init.h"
#endif

#include "implicit.h"
#include "elliptic.h"

#include "catch2/catch_all.hpp"

template< class Geometry, class Matrix, class Container>
struct Diffusion
{
    Diffusion( Geometry& g, double nu, unsigned order)
    {
        m_temp[0] = dg::evaluate( dg::zero, g);
        m_temp[1] = dg::evaluate( dg::zero, g);
        enum dg::direction dir = dg::str2direction( "centered");
        m_LaplacianM.construct( g, dir, 1);
        m_nu = nu;
        m_order = order;
    }
    void operator()(double, const Container& x, Container& y)
    {
        if( m_nu != 0)
        {
            dg::blas1::copy( x, m_temp[1]);
            for( unsigned p=0; p<m_order; p++)
            {
                using std::swap;
                swap( m_temp[0], m_temp[1]);
                dg::blas2::symv( m_nu, m_LaplacianM, m_temp[0], 0., m_temp[1]);
            }
            dg::blas1::axpby( -1., m_temp[1], 0., y);
        }
        else
            dg::blas1::scal( y, 0);
    }
    const Container& weights(){ return m_LaplacianM.weights();}
    const Container& precond(){ return m_LaplacianM.precond();}
  private:
    unsigned m_order;
    double m_nu;
    Container m_temp[2];
    dg::Elliptic<Geometry, Matrix,Container> m_LaplacianM;
};


inline const double alpha = -0.01;
inline const double nu = 1e-3;
inline const unsigned order = 2;
static double rhs( double x, double y) { return (1.-alpha*pow(2.*nu,order))*sin(x)*sin(y);}
static double sol(double x, double y)  { return sin( x)*sin(y);}
static double initial( double x, double y) {return rhs(x,y);}

// TODO Should we use a logging system for feltor algorithms?

TEST_CASE( "Default solver interface")
{
#ifdef WITH_MPI
    MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0}, {0,1});
#endif
    const double lx = M_PI;
    const double ly = 2.*M_PI;
    const dg::bc bcx = dg::DIR;
    const dg::bc bcy = dg::PER;

    unsigned n = 3, Nx = 48, Ny = 48;
    double eps = 1e-10;
    unsigned max_iter = 1000;

    INFO( "Computation on: "<< n <<" x "<< Nx <<" x "<< Ny);
	dg::x::CartesianGrid2d grid( 0, lx, 0, ly, n, Nx, Ny, bcx, bcy
#ifdef WITH_MPI
    , comm
#endif
    );
    dg::x::DVec w2d = dg::create::weights( grid);
    //create functions A(chi) x = b
    dg::x::DVec x =    dg::evaluate( initial, grid);
    const dg::x::DVec b =    dg::evaluate( rhs, grid);
    Diffusion<dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec> diff( grid, nu, order);

    //compute error
    const dg::x::DVec solution = dg::evaluate( sol, grid);
    const double norm = dg::blas2::dot( w2d, solution);
    dg::x::DVec error( solution);
    dg::exblas::udouble res;
    dg::DefaultSolver solver( diff, x, max_iter, eps);
    solver.set_benchmark(false);
    x  =    dg::evaluate( initial, grid);
    solver( alpha, 1., x, b);
    dg::blas1::axpby( 1.,x,-1., solution, error);
    double err = dg::blas2::dot( w2d, error);
    err = sqrt( err/norm); res.d = err;
    INFO( " Error "<<err << "\t"<<res.i);
    CHECK( err < 2e-7);
    // MW This does not succeed on github or msvc:
    // CHECK( abs( res.i - 4503686874401734415) < 2);
}

