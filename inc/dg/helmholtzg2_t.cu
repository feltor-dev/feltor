#include <iostream>

#include "blas.h"

#include "helmholtz.h"
#include "backend/xspacelib.cuh"
#include "backend/exceptions.h"
#include "multistep.h"
#include "cg.h"
#include "functors.h"

const double eps = 1e-8;
const double tau=1.0; 
const double alpha = -0.5*tau;
double lhs2( double x,double y){ return sin(x);}
// double lhs1( double x,double y){ return sin(x)*sin(y);}
double lhs1( double x,double y){ return sin(x);}
double rhs2( double x,double y){
    return  (-2.-2.*x+2.*cos(x)+2*x*cos(x)+sin(x)+2*x*sin(x)+x*x*sin(x)-2*alpha*sin(x)-2*x*alpha*sin(x)+alpha*alpha*sin(x))/(1.0+x)/alpha;
    
}
// double rhs1( double x,double y){ return  (1.-2.*(-0.5*tau))*sin(x)*sin(y);}
double rhs1( double x,double y){ return  (1.-alpha)*sin(x);}
// double dx2rhs2( double x,double y){ return (1.0 - 2.*alpha + alpha*alpha)*sin(x);}
double dx2rhs2( double x,double y){ return (1.+x)*sin(x)-2*alpha*sin(x)+alpha*alpha*(2*cos(x)/(1.+x)/(1.+x)-2*sin(x)/(1.+x)/(1.+x)/(1.+x)+sin(x)/(1.+x));}


int main()
{
    
    unsigned n, Nx, Ny, Nz; 
    std::cout << "Type n, Nx Ny and Nz\n";
    std::cin >> n>> Nx >> Ny >> Nz;
    dg::Grid2d grid2d( 0, 2.*M_PI, 0, 2.*M_PI, n, Nx, Ny,dg::DIR,dg::PER);
    const dg::DVec w2d = dg::create::weights( grid2d);
    const dg::DVec v2d = dg::create::inv_weights( grid2d);
    const dg::DVec one = dg::evaluate( dg::one, grid2d);
     dg::DVec rho = dg::evaluate( rhs2, grid2d);
    const dg::DVec rho1 = dg::evaluate( rhs1, grid2d);
    dg::DVec rholap = dg::evaluate( dx2rhs2, grid2d);
    const dg::DVec sol = dg::evaluate( lhs2, grid2d);
    const dg::DVec sol1 = dg::evaluate( lhs1, grid2d);
    dg::DVec x(rho.size(), 0.), rho_(rho);

    const dg::DVec chi = dg::evaluate( dg::LinearX(1.0,1.0), grid2d);
    
    dg::Helmholtz< dg::CartesianGrid2d, dg::DMatrix, dg::DVec > gamma1inv(  grid2d,grid2d.bcx(),grid2d.bcy(), alpha ,dg::centered);
    dg::Helmholtz2< dg::CartesianGrid2d, dg::DMatrix, dg::DVec > gamma2inv(  grid2d,grid2d.bcx(),grid2d.bcy(), alpha,dg::centered);
    dg::Elliptic< dg::CartesianGrid2d, dg::DMatrix, dg::DVec > lapperp(grid2d,grid2d.bcx(), grid2d.bcy(), dg::normed, dg::centered);
    gamma2inv.set_chi(chi); 
    


    dg::DVec x_(rho.size(), 0.);
    dg::Invert<dg::DVec> invert( x_, grid2d.size(), eps);
    dg::blas2::gemv(lapperp,rho,rholap); //lambda = - nabla_perp^2 phi
    dg::blas1::scal(rholap,alpha); // lambda = 0.5*tau_i*nabla_perp^2 phi
    
//test gamma2    
    unsigned number = invert( gamma2inv, x_, rholap);
            if(  number == invert.get_max())
            throw dg::Fail( eps);

    //Evaluation
    dg::blas1::axpby( 1., sol, -1., x_);

    std::cout << "number of iterations:  "<<number<<std::endl;
    std::cout << "abs error " << sqrt( dg::blas2::dot( w2d, x_))<<std::endl;
    std::cout << "rel error " << sqrt( dg::blas2::dot( w2d, x_)/ dg::blas2::dot( w2d, sol))<<std::endl;

    number = invert( gamma1inv, x_, rho1);
            if(  number == invert.get_max())
            throw dg::Fail( eps);
//test gamma 1
    //Evaluation
    dg::blas1::axpby( 1., sol1, -1., x_);

    std::cout << "number of iterations:  "<<number<<std::endl;
    std::cout << "abs error " << sqrt( dg::blas2::dot( w2d, x_))<<std::endl;
    std::cout << "rel error " << sqrt( dg::blas2::dot( w2d, x_)/ dg::blas2::dot( w2d, sol1))<<std::endl;

    
/*    
    std::cout << "Test 3d cylincdrical norm:\n";
    dg::Grid3d g3d( R_0, R_0+lx, 0, ly, 0,lz, n, Nx, Ny,Nz, bcx, dg::PER, dg::PER, dg::cylindrical);
    dg::DVec fct_ = dg::evaluate(fct, g3d );
    dg::DVec laplace_fct_ = dg::evaluate( laplace_fct, g3d);
    dg::DVec helmholtz_fct_ = dg::evaluate( helmholtz_fct, g3d);
    dg::DVec temp_(fct_);
    dg::Elliptic< dg::DMatrix, dg::DVec > laplaceM( g3d, dg::normed);
    dg::Helmholtz< dg::DMatrix, dg::DVec > helmholtz( g3d, alpha);
    dg::blas2::symv( laplaceM, fct_, temp_);
    dg::blas1::axpby( 1., laplace_fct_, -1., temp_);
    std::cout << "error Laplace " << sqrt( dg::blas2::dot( laplaceM.weights(), temp_))<<" (Note the supraconvergence!)"<<std::endl;
    dg::blas2::symv( helmholtz, fct_, temp_);
    dg::blas2::symv( helmholtz.precond(), temp_, temp_);
    dg::blas1::axpby( 1., helmholtz_fct_, -1, temp_);
    std::cout << "error " << sqrt( dg::blas2::dot( helmholtz.weights(), temp_))<<" (Note the supraconvergence!)"<<std::endl;*/



    //Tests GAMMA2

    return 0;
}



