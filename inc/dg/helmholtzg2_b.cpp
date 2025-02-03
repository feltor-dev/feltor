#include <iostream>

#include "blas.h"
#include "backend/timer.h"
#include "backend/typedefs.h"
#include "backend/exceptions.h"

#include "helmholtz.h"
#include "pcg.h"
#include "functors.h"

const double eps = 1e-4;
const double tau=1.0;
const double alpha = -0.5*tau;
double lhs( double x,double y){ return sin(x);}
double rhs( double x,double y){
    return  (-2.-2.*x+2.*cos(x)+2*x*cos(x)+sin(x)+2*x*sin(x)+x*x*sin(x)
    -2*alpha*sin(x)-2*x*alpha*sin(x)+alpha*alpha*sin(x))/(1.0+x)/alpha;

}
// double dxrhs( double x,double y){ return (1.0 - 2.*alpha + alpha*alpha)*sin(x);} //// chi=1
double dxrhs( double x,double y){ return (1.+x)*sin(x)-2*alpha*sin(x)+alpha*alpha*(2*cos(x)/(1.+x)/(1.+x)-2*sin(x)/(1.+x)/(1.+x)/(1.+x)+sin(x)/(1.+x));} // chi=x


int main()
{

    unsigned n, Nx, Ny;
    std::cout << "Type n, Nx and Ny\n";
    std::cin >> n>> Nx >> Ny;
    dg::Grid2d grid2d( 0, 2.*M_PI, 0, 2.*M_PI, n, Nx, Ny,dg::DIR,dg::PER);
    const dg::DVec w2d = dg::create::weights( grid2d);
    const dg::DVec v2d = dg::create::inv_weights( grid2d);
    const dg::DVec one = dg::evaluate( dg::one, grid2d);
    const dg::DVec sol = dg::evaluate( lhs, grid2d);
    dg::DVec x(sol.size(), 0.);
    const dg::DVec chi = dg::evaluate( dg::LinearX(1.0,1.0), grid2d);

    dg::Helmholtz2< dg::CartesianGrid2d, dg::DMatrix, dg::DVec > gamma2barinv(grid2d, alpha,dg::centered);
    dg::Elliptic< dg::CartesianGrid2d, dg::DMatrix, dg::DVec > gamma2tilde(grid2d, dg::centered);
    gamma2barinv.set_chi(chi);

    dg::PCG<dg::DVec> pcg2( x, grid2d.size());
    dg::PCG<dg::DVec> pcg1( x, grid2d.size());

    dg::DVec rho = dg::evaluate( rhs, grid2d);
    dg::DVec rholap = dg::evaluate( dxrhs, grid2d);
    dg::blas2::gemv(gamma2tilde,rho,rholap); //lambda = - nabla_perp^2 phi
    dg::blas1::scal(rholap,alpha); // lambda = 0.5*tau_i*nabla_perp^2 phi

    //test gamma2
    dg::Timer t;
    t.tic();
    unsigned number = pcg2.solve( gamma2barinv, x, rholap, 1., w2d, eps);
    t.toc();

    //Evaluation
    dg::blas1::axpby( 1., sol, -1., x);

    std::cout << "number of iterations:  "<<number<<std::endl;
    std::cout << "abs error " << sqrt( dg::blas2::dot( w2d, x))<<std::endl;
    std::cout << "rel error " << sqrt( dg::blas2::dot( w2d, x)/ dg::blas2::dot( w2d, sol))<<std::endl;
    std::cout << "took  " << t.diff()<<"s"<<std::endl;

    dg::DVec phi(x.size(), 0.);
    dg::blas1::scal(x,0.); //x=0
    //![doxygen]
    std::cout << "Alternative test with two Helmholtz operators\n";
    dg::Helmholtz< dg::CartesianGrid2d, dg::DMatrix, dg::DVec > gamma1inv(alpha, {grid2d, dg::centered});
    gamma1inv.set_chi( chi);
    dg::PCG<dg::DVec> pcgO(  x, grid2d.size());
    dg::PCG<dg::DVec> pcgOO( x, grid2d.size());
    t.tic();
    unsigned number1 = pcgO.solve( gamma1inv, phi, rholap, 1., w2d, eps/100);
    dg::blas1::pointwiseDot( phi, chi, phi);
    unsigned number2 = pcgOO.solve( gamma1inv, x, phi, 1., w2d, eps/100);
    t.toc();
    //Evaluation
    dg::blas1::axpby( 1., sol, -1., x);
    //![doxygen]

    std::cout << "number of iterations:  "<<number1<<" and "<<number2<<std::endl;
    std::cout << "abs error " << sqrt( dg::blas2::dot( w2d, x))<<std::endl;
    std::cout << "rel error " << sqrt( dg::blas2::dot( w2d, x)/ dg::blas2::dot( w2d, sol))<<std::endl;
    std::cout << "took  " << t.diff()<<"s"<<std::endl;

    return 0;
}
