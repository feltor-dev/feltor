#include <iostream>
#include "blas.h"

#include "helmholtz.h"
#include "polarization.h"
#include "multigrid.h"
#include "backend/exceptions.h"
#include "multistep.h"
#include "cg.h"
#include "functors.h"
#include "andersonacc.h"

const double eps = 1e-14;
const double tau = 1.;
const double alpha = -tau;
const double m = 4.;
const double n = 4.;

double phi_sol( double x,double y){ return sin(m*x)*sin(n*y);}
double rho_sol( double x,double y){ return (m*m+n*n)/(-1.+(m*m+n*n)*alpha)*sin(m*x)*sin(n*y);}


double amp = 0.9;
double pol( double x, double y) {return 1. + amp*sin(x)*sin(y); } //must be strictly positive
// double rho_sol_FF( double x, double y) { return (amp*cos(2.*x)*(1./sqrt(1.-4.*alpha) - 2.*cos(2.*y)/sqrt(1.-8.*alpha)) + amp* cos(2.*y)/sqrt(1.-4.*alpha)-(4.*sin(x)*sin(y))/sqrt(1.-2.*alpha))/2.; }
// double phi_sol_FF(double x, double y)  { return (sin(x)*sin(y))*sqrt(1.-(1.+1.)*alpha);}

double rho_sol_FF( double x, double y) { return (amp*cos(2.*x)*(1./sqrt(1.-4.*alpha) - 2.*cos(2.*y)/sqrt(1.-8.*alpha)) + amp* cos(2.*y)/sqrt(1.-4.*alpha)-(4.*sin(x)*sin(y))/sqrt(1.-2.*alpha))/2./sqrt(1.-(1.+1.)*alpha); }
double phi_sol_FF(double x, double y)  { return (sin(x)*sin(y));}

double rho_sol_FF_cold( double x, double y) { return 2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);}
double phi_sol_FF_cold(double x, double y)  { return sin( x)*sin(y);}

using DiaMatrix = cusp::dia_matrix<int, double, cusp::device_memory>;
using CooMatrix = cusp::coo_matrix<int, double, cusp::device_memory>;
using Matrix = dg::DMatrix;
using Container = dg::DVec;
using SubContainer = dg::DVec;

int main()
{
    dg::Timer t;

    unsigned n, Nx, Ny;
    std::cout << "Type n, Nx and Ny\n";
    std::cin >> n>> Nx >> Ny;
    dg::Grid2d grid2d( 0, 2.*M_PI, 0, 2.*M_PI, n, Nx, Ny,dg::DIR,dg::PER);
    const Container w2d = dg::create::weights( grid2d);
    const Container v2d = dg::create::inv_weights( grid2d);
    const Container one = dg::evaluate( dg::one, grid2d);
    const Container rho = dg::evaluate( rho_sol, grid2d);
    const Container sol = dg::evaluate( phi_sol, grid2d);
    Container rho_FF = dg::evaluate( rho_sol_FF, grid2d);
    const Container sol_FF = dg::evaluate( phi_sol_FF, grid2d);
    Container x(rho.size(), 0.), temp(rho), error(rho), x_gamma(x);
    const Container chi =  dg::evaluate( pol, grid2d);
    
    exblas::udouble res;
    unsigned number = 0;
    dg::Helmholtz< dg::CartesianGrid2d, Matrix, Container > gamma0inv(  grid2d,grid2d.bcx(),grid2d.bcy(), alpha ,dg::centered);
    dg::Elliptic< dg::CartesianGrid2d, Matrix, Container > lapperp(grid2d,grid2d.bcx(), grid2d.bcy(), dg::not_normed, dg::centered);

    dg::Invert<Container> invert( x, grid2d.size(), eps);
    double eps_gamma = 1e-14;
    dg::Polarization< dg::CartesianGrid2d, Matrix, DiaMatrix, CooMatrix, Container, SubContainer > polarization;
    
    polarization.construct(alpha, {eps_gamma, 0.1*eps_gamma, 0.1*eps_gamma}, grid2d, grid2d.bcx(), grid2d.bcy(), dg::not_normed, dg::centered, 1., false);

    
//     /////////////////delta-F polarization
//     //implementation with nested inversion (note: eps_gamma < eps_pol to have convergence)    
//     dg::blas1::scal(x,0.0);   
//     t.tic();
//     number = invert( polarization, x, rho);
//             if(  number == invert.get_max())
//             throw dg::Fail( eps);
//     dg::blas1::scal(x,-1.0);
//     t.toc();
//     
//     dg::blas1::axpby( 1., sol, -1., x, error);
//     res.d = sqrt( dg::blas2::dot( w2d, error));
//     std::cout << " Time: "<<t.diff() << "\n";
//     std::cout << "number of iterations:  "<<number<<std::endl;
//     std::cout << "abs error " << res.d<<"\t"<<res.i<<std::endl;
//     std::cout << "rel error " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, sol))<<std::endl;
//     
//     //implementation without nested inversion (faster and same error)
//     dg::blas1::scal(x,0.0);
//     t.tic();
//     dg::blas2::symv(gamma0inv, rho, temp); 
//     dg::blas1::pointwiseDot(v2d,temp,temp);
//     dg::blas1::scal(temp,-1.0);
//     number = invert( lapperp, x, temp);
//             if(  number == invert.get_max())
//             throw dg::Fail( eps);
//     t.toc();
//     
//     dg::blas1::axpby( 1., sol, -1., x, error);
// 
//     res.d = sqrt( dg::blas2::dot( w2d, error));
//     
//     std::cout << " Time: "<<t.diff() << "\n";
//     std::cout << "number of iterations:  "<<number<<std::endl;
//     std::cout << "abs error " << res.d<<"\t"<<res.i<<std::endl;
//     std::cout << "rel error " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, sol))<<std::endl;
//     
//     
// //     /////////////////Full-F polarization (order 2)    //TODO relative error does not converge , cauchy invert does not converge for some reason
// //     
//     lapperp.set_chi( chi);
// //     KrylovSqrtCauchySolve< dg::CartesianGrid2d, Matrix, DiaMatrix, CooMatrix, Container, SubContainer> sqrtsolve;
// //     sqrtsolve.construct( gamma0inv, grid2d, chi,  1e-12, 200, 20,  1e-12);
//     KrylovSqrtODESolve< dg::CartesianGrid2d, Matrix, DiaMatrix, CooMatrix, Container, SubContainer> sqrtsolve;
//     sqrtsolve.construct( gamma0inv, grid2d, chi,  1e-12,  1e-8, 1e-12, 200,  1e-12);
//     
//     dg::blas1::scal(x_gamma, 0.0);
//     dg::blas1::scal(temp, 0.0);
//     dg::blas1::scal(x, 0.0);
//     t.tic();
//     if (tau == 0) dg::blas1::copy(rho_FF,temp);
//     else sqrtsolve(rho_FF, temp); 
//     dg::blas1::scal(temp,-1.0);
//     number = invert( lapperp, x_gamma, temp);
//             if(  number == invert.get_max())
//             throw dg::Fail( eps);
//     if (tau == 0) dg::blas1::copy(x_gamma, x);
//     else sqrtsolve(x_gamma, x);       
//     t.toc();
//     dg::blas1::axpby( 1., sol_FF, -1., x, error);
// 
//     res.d = sqrt( dg::blas2::dot( w2d, error));
//     
//     std::cout << " Time: "<<t.diff() << "\n";
//     std::cout << "number of iterations:  "<<number<<std::endl;
//     std::cout << "abs error " << res.d<<"\t"<<res.i<<std::endl;
//     std::cout << "rel error " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, sol_FF))<<std::endl;
//     
    
    //testing initialization //TODO
    dg::PolarizationN< dg::CartesianGrid2d, Matrix, Container > polN(grid2d,grid2d.bcx(), grid2d.bcy(), dg::normed, dg::centered);
    
    Container rho_cold = dg::evaluate(rho_sol_FF_cold, grid2d);
    Container phi_cold = dg::evaluate(phi_sol_FF_cold, grid2d);
    
    dg::AndersonAcceleration<Container> acc( x, 10);
    
    polN.set_phi(phi_cold);
    dg::blas1::scal(x, 0.0);
    dg::blas1::plus(x, 1.0); //must be greater than 0
    
//     dg::blas1::copy(chi, x); //solution as guess
    const double eps = 1e-3;
    double damping;
    unsigned restart;
    std::cout << "Type damping (1e-9) and restart (10) \n";
    std::cin >> damping >> restart;
    std::cout << "Number of iterations "<< acc.solve( polN, x, rho_cold, w2d, eps, eps*eps, grid2d.size()*grid2d.size(), damping,restart, true)<<std::endl;
    dg::blas1::axpby( 1., chi, -1., x, error);

    res.d = sqrt( dg::blas2::dot( w2d, error));
    
    std::cout << "rel error " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, chi))<<std::endl;
    
    return 0;
}
