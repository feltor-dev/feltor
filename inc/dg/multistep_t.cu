#include <iostream>

#include "multistep.h"
#include "elliptic.h"

template < class Matrix, class container = thrust::device_vector<double> >
struct RHS
{
    typedef container Vector;
    RHS( const dg::Grid2d& g, double D): D_(D), laplaceM(g, dg::normed)
    { }
    void operator()( const std::vector<container>& y, std::vector<container>& yp)
    {
        dg::blas1::axpby( 0., y, 0., yp);
    }
  private:
    double D_;
    dg::Elliptic<dg::CartesianGrid2d, Matrix, container> laplaceM;
};

template< class Matrix, class container>
struct Diffusion
{
    Diffusion( const dg::Grid2d& g, double nu): nu_(nu),
        w2d( dg::create::weights(g)), 
        v2d( dg::create::inv_weights(g)),
        LaplacianM( g, dg::normed) 
        { }

    void operator()( const std::vector<container>& x, std::vector<container>& y)
    {
        for(unsigned i=0; i<x.size(); i++)
        {
            dg::blas2::gemv( LaplacianM, x[i], y[i]);
        }
        dg::blas1::axpby( 0.,y, -nu_, y);
    }
    const container& inv_weights(){return v2d;}
    const container& weights(){return w2d;}
    const container& precond(){return v2d;}
  private:
    double nu_;
    const container w2d, v2d;
    dg::Elliptic<dg::CartesianGrid2d, Matrix, container> LaplacianM;
};


const unsigned n = 3;
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;

const unsigned k = 3;
const double nu = 0.01;
const double T = 1.0;
//const unsigned NT = (unsigned)(nu*T*n*n*N*N/0.01/lx/lx);

double sine( double x, double y) {return sin(x)*sin(y);}
double sol( double x, double y) {return exp( -2.*nu*T)*sine(x, y);}


int main()
{
    double dt, NT, eps;
    unsigned Nx, Ny;
    std::cout << "Type Nx (20), Ny (20) and timestep (0.1) and eps( 1e-8)!\n";
    std::cin >> Nx >> Ny >> dt >> eps;
    NT = (unsigned)(T/dt);

    std::cout << "Test Karniadakis scheme on diffusion equation\n";
    std::cout << "RK order K:               "<< k <<std::endl;
    std::cout << "Number of gridpoints:     "<<Nx*Ny<<std::endl;
    std::cout << "# of timesteps:           "<<NT<<std::endl;

    dg::Grid2d grid( 0, lx, 0, ly, n, Nx, Ny, dg::PER, dg::PER);
    dg::DVec w2d = dg::create::weights( grid);

    std::vector<dg::DVec> y0(2, dg::evaluate( sine, grid)), y1(y0);

    RHS<dg::DMatrix, dg::DVec> rhs( grid, nu);
    Diffusion<dg::DMatrix, dg::DVec> diffusion( grid, nu);
    dg::Karniadakis< std::vector<dg::DVec> > tvb( y0, y0[0].size(), eps);
    tvb.init( rhs, diffusion, y0, dt);
    dg::SIRK< std::vector<dg::DVec> > sirk( y0, y0[0].size(), eps);

    //thrust::swap(y0, y1);
    for( unsigned i=0; i<NT; i++)
    {
        tvb( rhs, diffusion, y0);
        //sirk( rhs, diffusion, y0, y1, dt);
        y0.swap(y1);
    }
    double norm_y0 = dg::blas2::dot( w2d, y0[0]);
    std::cout << "Normalized y0 after "<< NT <<" steps is "<< norm_y0 << std::endl;
    dg::DVec solution = dg::evaluate( sol, grid), error( solution);
    double norm_sol = dg::blas2::dot( w2d, solution);
    dg::blas1::axpby( -1., y0[0], 1., error);
    std::cout << "Normalized solution is "<<  norm_sol<< std::endl;
    double norm_error = dg::blas2::dot( w2d, error);
    std::cout << "Relative error is      "<< sqrt( norm_error/norm_sol)<<" (0.0020084 Karniadakis) (0.000148647 SIRK)\n";
    //n = 1 -> p = 1 (Sprung in laplace macht n=1 eine Ordng schlechter) 
    //n = 2 -> p = 2
    //n = 3 -> p = 3
    //n = 4 -> p = 4
    //n = 5 -> p = 5

    return 0;
}
