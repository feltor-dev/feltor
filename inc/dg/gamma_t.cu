#include <iostream>

#include "blas.h"

#include "gamma.cuh"
#include "xspacelib.cuh"
#include "karniadakis.cuh"

#include "cg.cuh"
template< class container>
struct Diffusion
{
    Diffusion( const dg::Grid2d<double>& g, double nu):
        w2d( dg::create::w2d( g)), v2d( dg::create::v2d(g)) { 
        dg::Matrix Laplacian_ = dg::create::laplacianM( g, dg::normed, dg::XSPACE); 
        cusp::blas::scal( Laplacian_.values, -nu);
        Laplacian = Laplacian_;
        }
    void operator()( const container& x, container& y)
    {
        //dg::blas1::axpby( 0., x, 0., y);
        dg::blas2::gemv( Laplacian, x, y);
    }
    const container& weights(){return w2d;}
    const container& precond(){return v2d;}
  private:
    const container w2d, v2d;
    dg::DMatrix Laplacian;
};

const double eps = 1e-4;
const double alpha = -0.5; 
double lhs( double x, double y){ return sin(x)*sin(y);}
double rhs( double x, double y){ return (1.-2.*alpha)*sin(x)*sin(y);}
//double rhs( double x, double y){ return lhs(x,y);}
int main()
{
    
    unsigned n, Nx, Ny; 
    std::cout << "Type n, Nx and Ny\n";
    std::cin >> n>> Nx >> Ny;
    dg::Grid2d<double> grid( 0, 2.*M_PI, 0, 2.*M_PI, n, Nx, Ny, dg::DIR, dg::PER);
    const dg::DVec w2d = dg::create::w2d( grid);
    const dg::DVec v2d = dg::create::v2d( grid);
    dg::DVec rho = dg::evaluate( rhs, grid);
    const dg::DVec sol = dg::evaluate( lhs, grid);
    dg::DVec x(rho.size(), 0.);
    //dg::DVec x(rho);

    dg::DMatrix A = dg::create::laplacianM( grid, dg::normed, dg::XSPACE); 
    dg::Gamma< dg::DMatrix, dg::DVec > gamma1( A, w2d, alpha);
    dg::Helmholtz2d <dg::DVec> diff( grid, alpha, eps);

    dg::CG< dg::DVec > cg(x, x.size());
    dg::blas2::symv( w2d, rho, rho);
    unsigned number = cg( gamma1, x, rho, v2d, eps);
    dg::DVec x_(rho.size(), 0.);
    rho = dg::evaluate( rhs, grid);
    diff( rho, x_);

    dg::DVec x__(rho.size(), 0.);
    rho = dg::evaluate( rhs, grid);
    Diffusion<dg::DVec> diffusion( grid, 1.);
    dg::detail::Implicit<Diffusion<dg::DVec> > implicit( alpha, diffusion);
    dg::blas2::symv( diffusion.weights(), rho, rho);
    number = cg( implicit, x__, rho, diffusion.precond(), eps);
    dg::blas1::axpby( 1., sol, -1., x);
    dg::blas1::axpby( 1., sol, -1., x_);
    dg::blas1::axpby( 1., sol, -1., x__);

    std::cout << "number of iterations:  "<<number<<std::endl;
    std::cout << "error " << sqrt( dg::blas2::dot( w2d, x))<<std::endl;
    std::cout << "error " << sqrt( dg::blas2::dot( w2d, x_))<<std::endl;
    std::cout << "error " << sqrt( dg::blas2::dot( w2d, x__))<<std::endl;




    return 0;
}



