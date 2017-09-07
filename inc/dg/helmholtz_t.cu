#include <iostream>

#include "blas.h"

#include "helmholtz.h"
#include "backend/typedefs.cuh"
#include "multistep.h"

#include "cg.h"

//template< class container>
//struct Diffusion
//{
//    Diffusion( const dg::Grid2d& g, double nu):
//        nu_(nu),
//        w2d( dg::create::weights( g)), v2d( dg::create::inv_weights(g)) { 
//        dg::Matrix Laplacian_ = dg::create::laplacianM( g, dg::normed); 
//        cusp::blas::scal( Laplacian_.values, -nu);
//        Laplacian = Laplacian_;
//        }
//    void operator()( const container& x, container& y)
//    {
//        //dg::blas1::axpby( 0., x, 0., y);
//        dg::blas2::gemv( Laplacian, x, y);
//    }
//    const container& weights(){return w2d;}
//    const container& precond(){return v2d;}
//  private:
//    double nu_;
//    const container w2d, v2d;
//    dg::DMatrix Laplacian;
//};

const double eps = 1e-4;
const double alpha = -0.5; 
double lhs( double x, double y){ return sin(x)*sin(y);}
double rhs( double x, double y){ return (1.-2.*alpha)*sin(x)*sin(y);}
const double R_0 = 1000;
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
const double lz = 2.*M_PI;
double fct(double x, double y, double z){ return sin(x-R_0)*sin(y);}
double laplace_fct( double x, double y, double z) { 
    return -1./x*cos(x-R_0)*sin(y) + 2.*sin(x-R_0)*sin(y);}
double helmholtz_fct( double x, double y, double z) { 
    return fct(x,y,z) - alpha*laplace_fct(x,y,z);}

dg::bc bcx = dg::DIR;
double initial( double x, double y, double z) {return sin(0);}
int main()
{
    
    unsigned n, Nx, Ny, Nz; 
    std::cout << "Type n, Nx Ny and Nz\n";
    std::cin >> n>> Nx >> Ny >> Nz;
    dg::Grid2d grid( 0, 2.*M_PI, 0, 2.*M_PI, n, Nx, Ny, dg::DIR, dg::PER);
    const dg::DVec w2d = dg::create::weights( grid);
    const dg::DVec v2d = dg::create::inv_weights( grid);
    const dg::DVec rho = dg::evaluate( rhs, grid);
    const dg::DVec sol = dg::evaluate( lhs, grid);
    dg::DVec x(rho.size(), 0.), rho_(rho);

    dg::Helmholtz<dg::CartesianGrid2d, dg::DMatrix, dg::DVec > gamma1inv( grid, alpha);

    std::cout << "FIRST METHOD:\n";
    dg::CG< dg::DVec > cg(x, x.size());
    dg::blas2::symv( w2d, rho, rho_);
    unsigned number = cg( gamma1inv, x, rho_, v2d, eps);

    std::cout << "SECOND METHOD:\n";
    dg::DVec x_(rho.size(), 0.);
    dg::Invert<dg::DVec> invert( x_, grid.size(), eps);
    dg::Helmholtz< dg::CartesianGrid2d, dg::DMatrix, dg::DVec > maxwell( grid, alpha);
    invert( maxwell, x_, rho);

    //std::cout << "THIRD METHOD:\n";
    //dg::DVec x__(rho.size(), 0.);
    //Diffusion<dg::DVec> diffusion( grid, 1.);
    //dg::DVec temp (w2d);
    //dg::detail::Implicit<Diffusion<dg::DVec>, dg::DVec > implicit( alpha, diffusion,temp);
    //dg::blas2::symv( diffusion.weights(), rho, rho_);
    //number = cg( implicit, x__, rho_, diffusion.precond(), eps);

    //Evaluation
    dg::blas1::axpby( 1., sol, -1., x);
    dg::blas1::axpby( 1., sol, -1., x_);
    //dg::blas1::axpby( 1., sol, -1., x__);

    std::cout << "number of iterations:  "<<number<<std::endl;
    std::cout << "ALL METHODS SHOULD DO THE SAME!\n";
    std::cout << "error1 " << sqrt( dg::blas2::dot( w2d, x))<<std::endl;
    std::cout << "error2 " << sqrt( dg::blas2::dot( w2d, x_))<<std::endl;
    //std::cout << "error3 " << sqrt( dg::blas2::dot( w2d, x__))<<std::endl;
    std::cout << "Test 3d cylincdrical norm:\n";
    dg::CylindricalGrid3d g3d( R_0, R_0+lx, 0, ly, 0,lz, n, Nx, Ny,Nz, bcx, dg::PER, dg::PER);
    dg::DVec fct_ = dg::evaluate(fct, g3d );
    dg::DVec laplace_fct_ = dg::evaluate( laplace_fct, g3d);
    dg::DVec helmholtz_fct_ = dg::evaluate( helmholtz_fct, g3d);
    dg::DVec temp_(fct_);
    dg::Elliptic< dg::CylindricalGrid3d, dg::DMatrix, dg::DVec > laplaceM( g3d, dg::normed);
    dg::Helmholtz< dg::CylindricalGrid3d, dg::DMatrix, dg::DVec > helmholtz( g3d, alpha);
    dg::blas2::symv( laplaceM, fct_, temp_);
    dg::blas1::axpby( 1., laplace_fct_, -1., temp_);
    dg::DVec w3d =  laplaceM.inv_weights();
    dg::blas1::transform(w3d, w3d,dg::INVERT<double>());
    std::cout << "error Laplace " << sqrt( dg::blas2::dot( w3d, temp_))<<" (Note the supraconvergence!)"<<std::endl;
    dg::blas2::symv( helmholtz, fct_, temp_);
    dg::blas1::pointwiseDot( helmholtz.inv_weights(), temp_, temp_);
    dg::blas1::axpby( 1., helmholtz_fct_, -1, temp_);
    std::cout << "error " << sqrt( dg::blas2::dot( w3d, temp_))<<" (Note the supraconvergence!)"<<std::endl;




    return 0;
}



