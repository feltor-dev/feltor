#include <iostream>

#include "blas.h"

#include "helmholtz.h"
#include "backend/xspacelib.cuh"
#include "multistep.h"

#include "cg.h"

template< class container>
struct Diffusion
{
    Diffusion( const dg::Grid2d<double>& g, double nu):
        w2d( dg::create::weights( g)), v2d( dg::create::inv_weights(g)) { 
        dg::Matrix Laplacian_ = dg::create::laplacianM( g, dg::normed); 
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
    const dg::DVec w2d = dg::create::weights( grid);
    const dg::DVec v2d = dg::create::inv_weights( grid);
    const dg::DVec rho = dg::evaluate( rhs, grid);
    const dg::DVec sol = dg::evaluate( lhs, grid);
    dg::DVec x(rho.size(), 0.), rho_(rho);

    dg::Helmholtz< dg::DMatrix, dg::DVec, dg::DVec > gamma1inv( grid, alpha);

    std::cout << "FIRST METHOD:\n";
    dg::CG< dg::DVec > cg(x, x.size());
    dg::blas2::symv( w2d, rho, rho_);
    unsigned number = cg( gamma1inv, x, rho_, v2d, eps);

    std::cout << "SECOND METHOD:\n";
    dg::DVec x_(rho.size(), 0.);
    dg::Invert<dg::DVec> invert( x_, grid.size(), eps);
    dg::Helmholtz< dg::DMatrix, dg::DVec, dg::DVec > maxwell( grid, alpha);
    invert( maxwell, x_, rho);

    std::cout << "THIRD METHOD:\n";
    dg::DVec x__(rho.size(), 0.);
    Diffusion<dg::DVec> diffusion( grid, 1.);
    dg::DVec temp (w2d);
    dg::detail::Implicit<Diffusion<dg::DVec>, dg::DVec > implicit( alpha, diffusion,temp);
    dg::blas2::symv( diffusion.weights(), rho, rho_);
    number = cg( implicit, x__, rho_, diffusion.precond(), eps);

    //Evaluation
    dg::blas1::axpby( 1., sol, -1., x);
    dg::blas1::axpby( 1., sol, -1., x_);
    dg::blas1::axpby( 1., sol, -1., x__);

    std::cout << "number of iterations:  "<<number<<std::endl;
    std::cout << "ALL METHODS SHOULD DO THE SAME!\n";
    std::cout << "error1 " << sqrt( dg::blas2::dot( w2d, x))<<std::endl;
    std::cout << "error2 " << sqrt( dg::blas2::dot( w2d, x_))<<std::endl;
    std::cout << "error3 " << sqrt( dg::blas2::dot( w2d, x__))<<std::endl;




    return 0;
}



