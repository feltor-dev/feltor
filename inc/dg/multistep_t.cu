#include <iostream>

#include <cusp/ell_matrix.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "multistep.h"
#include "backend/grid.h"
#include "backend/evaluation.cuh"
#include "backend/derivatives.cuh"
#include "backend/typedefs.cuh"

template < class container = thrust::device_vector<double> >
struct RHS
{
    typedef container Vector;
    typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    RHS( const dg::Grid2d<double>& g, double D): D_(D) 
    {
        laplaceM = dg::create::laplacianM( g, dg::normed);
    }
    void operator()( const std::vector<container>& y, std::vector<container>& yp)
    {
        dg::blas1::axpby( 0., y, 0., yp);
    }
  private:
    double D_;
    cusp::ell_matrix<int, double, MemorySpace> laplaceM;
};

template< class container>
struct Diffusion
{
    Diffusion( const dg::Grid2d<double>& g, double nu): nu_(nu),
        w2d(dg::create::weights( g)), v2d(dg::create::inv_weights(g)) { 
        LaplacianM = dg::create::laplacianM( g, dg::normed); 
        }

    void operator()( const std::vector<container>& x, std::vector<container>& y)
    {
        for(unsigned i=0; i<x.size(); i++)
        {
            dg::blas2::gemv( LaplacianM, x[i], y[i]);
        }
        dg::blas1::axpby( 0.,y, -nu_, y);
    }
    const container& weights(){return w2d;}
    const container& precond(){return v2d;}
  private:
    double nu_;
    const container w2d, v2d;
    dg::DMatrix LaplacianM;
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


using namespace std;
using namespace dg;

int main()
{
    double dt, NT, eps;
    unsigned Nx, Ny;
    cout << "Type Nx (20), Ny (20) and timestep (0.1) and eps( 1e-8)!\n";
    cin >> Nx >> Ny >> dt >> eps;
    NT = (unsigned)(T/dt);

    cout << "Test Karniadakis scheme on diffusion equation\n";
    cout << "RK order K:               "<< k <<endl;
    cout << "Number of gridpoints:     "<<Nx*Ny<<endl;
    cout << "# of timesteps:           "<<NT<<endl;

    Grid2d<double> grid( 0, lx, 0, ly, n, Nx, Ny, PER, PER);
    dg::DVec w2d = create::weights( grid);

    std::vector<DVec> y0(2, evaluate( sine, grid)), y1(y0);

    RHS<DVec> rhs( grid, nu);
    Diffusion<DVec> diffusion( grid, nu);
    dg::Karniadakis< std::vector<DVec> > tvb( y0, y0[0].size(), eps);
    tvb.init( rhs, diffusion, y0, dt);
    dg::SIRK< std::vector<dg::DVec> > sirk( y0, y0[0].size(), eps);

    //thrust::swap(y0, y1);
    for( unsigned i=0; i<NT; i++)
    {
        //tvb( rhs, diffusion, y0);
        sirk( rhs, diffusion, y0, y1, dt);
        y0.swap(y1);
    }
    double norm_y0 = blas2::dot( w2d, y0[0]);
    cout << "Normalized y0 after "<< NT <<" steps is "<< norm_y0 << endl;
    DVec solution = evaluate( sol, grid), error( solution);
    double norm_sol = blas2::dot( w2d, solution);
    blas1::axpby( -1., y0[0], 1., error);
    cout << "Normalized solution is "<<  norm_sol<< endl;
    double norm_error = blas2::dot( w2d, error);
    cout << "Relative error is      "<< sqrt( norm_error/norm_sol)<<" (0.000141704)\n";
    //n = 1 -> p = 1 (Sprung in laplace macht n=1 eine Ordng schlechter) 
    //n = 2 -> p = 2
    //n = 3 -> p = 3
    //n = 4 -> p = 4
    //n = 5 -> p = 5

    return 0;
}
