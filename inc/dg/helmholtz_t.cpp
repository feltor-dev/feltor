#include <iostream>

#include "blas.h"

#include "helmholtz.h"
#include "backend/typedefs.h"
#include "multistep.h"

#include "pcg.h"

#include "catch2/catch_test_macros.hpp"

inline const double tau=1.0;
inline const double alpha = -0.5*tau;
inline const double R_0 = 1000;
inline const double lx = 2.*M_PI;
inline const double ly = 2.*M_PI;
inline const double lz = 2.*M_PI;
inline dg::bc bcx = dg::DIR;

// 2d
static double lhs( double x, double y){ return sin(x)*sin(y);}
static double rhs( double x, double y){ return (1.-2.*alpha)*sin(x)*sin(y);}
// 3d Cylindrical grid
static double fct(double x, double y, double z){ return sin(x-R_0)*sin(y);}
static double laplace_fct( double x, double y, double z) {
    return -1./x*cos(x-R_0)*sin(y) + 2.*sin(x-R_0)*sin(y);}
static double helmholtz_fct( double x, double y, double z) {
    return fct(x,y,z) - alpha*laplace_fct(x,y,z);}

// Gamma 2
static double lhs2( double x,double y){ return sin(x);}
// double lhs1( double x,double y){ return sin(x)*sin(y);}
static double lhs1( double x,double y){ return sin(x);}
static double rhs2( double x,double y){
    return  (-2.-2.*x+2.*cos(x)+2*x*cos(x)+sin(x)+2*x*sin(x)+x*x*sin(x)-2*alpha*sin(x)-2*x*alpha*sin(x)+alpha*alpha*sin(x))/(1.0+x)/alpha;

}
// double rhs1( double x,double y){ return  (1.-2.*(-0.5*tau))*sin(x)*sin(y);}
static double rhs1( double x,double y){ return  (1.-alpha)*sin(x);}
// double dx2rhs2( double x,double y){ return (1.0 - 2.*alpha + alpha*alpha)*sin(x);}
static double dx2rhs2( double x,double y){ return (1.+x)*sin(x)-2*alpha*sin(x)+alpha*alpha*(2*cos(x)/(1.+x)/(1.+x)-2*sin(x)/(1.+x)/(1.+x)/(1.+x)+sin(x)/(1.+x));}


TEST_CASE( "Helmholtz")
{
    // TODO Maybe a convergence test is better
    SECTION( "2d Gamma 1 inverse")
    {
        const double eps = 1e-8;
        const unsigned n = 5, Nx = 21, Ny = 20;
        dg::x::Grid2d grid( 0, lx, 0, ly, n, Nx, Ny, bcx, dg::PER);
        const dg::x::DVec w2d = dg::create::weights( grid);
        const dg::x::DVec rho = dg::evaluate( rhs, grid);
        const dg::x::DVec sol = dg::evaluate( lhs, grid);
        dg::x::DVec x(rho.size(), 0.);

        dg::Helmholtz<dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec>
            gamma1inv( alpha, {grid});

        dg::PCG pcg(x, grid.size());
        unsigned number = pcg.solve( gamma1inv, x, rho, 1., w2d, eps);
        //Evaluation
        dg::blas1::axpby( 1., sol, -1., x);

        INFO( "number of iterations:  "<<number);
        double res = sqrt( dg::blas2::dot( w2d, x));
        INFO( "error " << res);
        CHECK( res < 2e-7);
    }
    SECTION( "2d Gamma 2 inverse")
    {
        const double eps = 1e-8;
        const unsigned n = 3, Nx = 15, Ny = 20;
        dg::x::Grid2d grid2d( 0, lx, 0, ly, n, Nx, Ny, bcx, dg::PER);
        const dg::x::DVec w2d = dg::create::weights( grid2d);
        dg::x::DVec rho = dg::evaluate( rhs2, grid2d);
        const dg::x::DVec rho1 = dg::evaluate( rhs1, grid2d);
        dg::x::DVec rholap = dg::evaluate( dx2rhs2, grid2d);
        const dg::x::DVec sol = dg::evaluate( lhs2, grid2d);
        const dg::x::DVec sol1 = dg::evaluate( lhs1, grid2d);
        dg::x::DVec x1(rho.size(), 0.), x2( x1);

        const dg::DVec chi = dg::evaluate( dg::LinearX(1.0,1.0), grid2d);

        dg::Helmholtz< dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec >
            gamma1inv( alpha, {grid2d, dg::centered});

        // Note that Helmholtz2 is deprecated
        dg::Helmholtz2< dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec >
            gamma2inv(  grid2d, grid2d.bcx(),grid2d.bcy(), alpha,dg::centered);
        dg::Elliptic< dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec >
            lapperp(grid2d,grid2d.bcx(), grid2d.bcy(), dg::centered);
        gamma2inv.set_chi(chi);

        dg::PCG pcg( x2, grid2d.size());
        dg::blas2::gemv(lapperp, rho, rholap); //lambda = - nabla_perp^2 phi
        dg::blas1::scal(rholap, alpha); // lambda = 0.5*tau_i*nabla_perp^2 phi

        //test gamma2
        unsigned number = pcg.solve( gamma2inv, x2, rholap, 1., w2d, eps);

        dg::blas1::axpby( 1., sol, -1., x2);
        double res = sqrt( dg::blas2::dot( w2d, x2));

        INFO( "number of iterations:  "<<number);
        INFO( "abs error " << res);
        res = sqrt( dg::blas2::dot( w2d, x2)/ dg::blas2::dot( w2d, sol));
        INFO( "rel error " << res);
        CHECK( res < 1e-4);

        number = pcg.solve( gamma1inv, x1, rho1, 1., w2d, eps);
        //test gamma 1
        dg::blas1::axpby( 1., sol1, -1., x1);
        res = sqrt( dg::blas2::dot( w2d, x1));

        INFO( "number of iterations:  "<<number);
        INFO( "abs error " << res);
        res = sqrt( dg::blas2::dot( w2d, x1)/ dg::blas2::dot( w2d, sol));
        INFO( "rel error " << res);
        CHECK( res < 1e-4);
    }
    SECTION( "Test direct 3d cylincdrical norm")
    {
        const unsigned n = 5, Nx = 21, Ny = 20, Nz = 10;
        dg::x::CylindricalGrid3d g3d( R_0, R_0+lx, 0, ly, 0,lz, n, Nx, Ny,Nz,
                bcx, dg::PER, dg::PER);
        const dg::x::DVec w3d = dg::create::weights( g3d);
        dg::x::DVec fctv           = dg::evaluate(fct, g3d );
        dg::x::DVec laplace_fctv   = dg::evaluate( laplace_fct, g3d);
        dg::x::DVec helmholtz_fctv = dg::evaluate( helmholtz_fct, g3d);
        dg::x::DVec temp(fctv);
        dg::Elliptic< dg::x::CylindricalGrid3d, dg::x::DMatrix, dg::x::DVec >
            laplaceM( g3d);
        dg::Helmholtz< dg::x::CylindricalGrid3d, dg::x::DMatrix, dg::x::DVec >
            helmholtz( alpha, {g3d});
        dg::blas2::symv( laplaceM, fctv, temp);
        dg::blas1::axpby( 1., laplace_fctv, -1., temp);
        double res = sqrt( dg::blas2::dot( w3d, temp));
        INFO( "Error Laplace " << res<<" (Note the supraconvergence!)");
        CHECK( res < 2e-3);
        dg::blas2::symv( helmholtz, fctv, temp);
        dg::blas1::axpby( 1., helmholtz_fctv, -1, temp);
        res = sqrt( dg::blas2::dot( w3d, temp));
        INFO( "error " << res<<" (Note the supraconvergence!)");
        CHECK( res < 1e-3);
    }


}
