#include <iostream>

#include <cusp/ell_matrix.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "rk.cuh"
#include "grid.cuh"
#include "evaluation.cuh"
#include "derivatives.cuh"
#include "preconditioner.cuh"

#include "blas.h"

template <class T, size_t n, class container = thrust::device_vector<T> >
struct RHS
{
    typedef container Vector;
    typedef typename thrust::iterator_space<typename container::iterator>::type MemorySpace;
    RHS( const dg::Grid<T,n>& g, T D):hx_(g.hx()), hy_(g.hy()), D_(D) 
    {
        laplace = dg::create::laplacian( g, dg::not_normed, dg::LSPACE);
    }
    void operator()( const container& y, container& yp)
    {
        dg::blas2::symv( laplace, y, yp);
        dg::blas2::symv( -D_, dg::T2D<T,n>(hx_, hy_), yp, 0., yp);
    }
  private:
    double hx_, hy_, D_;
    cusp::ell_matrix<int, T, MemorySpace> laplace;
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



//typedef thrust::device_vector<double> DVec;
typedef thrust::host_vector<double> DVec;
    

using namespace std;
using namespace dg;

int main()
{
    double dt, NT;
    unsigned Nx, Ny;
    cout << "Type Nx, Ny and timestep!\n";
    cin >> Nx >> Ny >> dt;
    NT = (unsigned)(T/dt);


    cout << "Test RK scheme on diffusion equation\n";
    cout << "Polynomial coefficients:  "<< n<<endl;
    cout << "RK order K:               "<< k <<endl;
    cout << "Number of gridpoints:     "<<Nx*Ny<<endl;
    cout << "# of timesteps:           "<<NT<<endl;

    Grid<double, n> grid( 0, lx, 0, ly ,Nx, Ny, DIR, DIR);
    S2D<double, n> s2d( grid.hx(), grid.hy());

    DVec y0 = expand( sine, grid), y1(y0);

    RHS<double,n, DVec> rhs( grid, nu);
    RK< k, DVec > rk( y0);
    AB< k, DVec > ab( y0);

    ab.init( rhs, y0, dt);
    //thrust::swap(y0, y1);
    for( unsigned i=0; i<NT; i++)
    {
        ab( rhs, y0, y1, dt);
        thrust::swap(y0, y1);
    }
    double norm_y0 = blas2::dot( s2d, y0);
    cout << "Normalized y0 after "<< NT <<" steps is "<< norm_y0 << endl;
    DVec solution = expand( sol, grid), error( solution);
    double norm_sol = blas2::dot( s2d, solution);
    blas1::axpby( -1., y0, 1., error);
    cout << "Normalized solution is "<<  norm_sol<< endl;
    double norm_error = blas2::dot( s2d, error);
    cout << "Relative error is      "<< sqrt( norm_error/norm_sol)<< endl;
    //n = 1 -> p = 1 (Sprung in laplace macht n=1 eine Ordng schlechter) 
    //n = 2 -> p = 2
    //n = 3 -> p = 3
    //n = 4 -> p = 4
    //n = 5 -> p = 5

    return 0;
}
