#include <iostream>

#include <cusp/ell_matrix.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "multistep.h"
#include "grid.h"
#include "helmholtz.h"
#include "evaluation.cuh"
#include "derivatives.cuh"
#include "typedefs.cuh"

#include "blas.h"

template < class container = thrust::device_vector<double> >
struct Identity
{
    Identity( const dg::Grid2d<double>& g): weights_(dg::create::weights(g)), precond_(dg::create::precond(g)){}
    void operator()( container& y, container& yp) const
    {
        dg::blas1::axpby( 0, y, 0, yp);
        dg::blas1::axpby( 0., y, 0., y); //destroy y
    }
    const container& weights(){return weights_;}
    const container& precond(){return precond_;}
    private:
    container weights_, precond_;

};
template < class container = thrust::device_vector<double> >
struct RHS
{
    typedef container Vector;
    typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    RHS( const dg::Grid2d<double>& g, double D): D_(D), weights_(dg::create::weights(g)), precond_(dg::create::precond(g))
    {
        laplaceM = dg::create::laplacianM( g, dg::normed);
    }
    void operator()( container& y, container& yp)
    {
        dg::blas2::symv( laplaceM, y, yp);
        dg::blas1::axpby( -D_, yp, 0., yp);
        dg::blas1::axpby( 0., y, 0., y); //destroy y
    }
    const container& weights(){return weights_;}
    const container& precond(){return precond_;}
  private:
    double D_;
    container weights_, precond_;
    cusp::ell_matrix<int, double, MemorySpace> laplaceM;
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
    double dt, NT;
    unsigned Nx, Ny;
    cout << "Type Nx (20), Ny (20) and timestep (0.01)!\n";
    cin >> Nx >> Ny >> dt;
    NT = (unsigned)(T/dt);


    cout << "Test RK scheme on diffusion equation\n";
    cout << "Polynomial coefficients:  "<< n<<endl;
    cout << "RK order K:               "<< k <<endl;
    cout << "Number of gridpoints:     "<<Nx*Ny<<endl;
    cout << "# of timesteps:           "<<NT<<endl;

    Grid2d<double> grid( 0, lx, 0, ly, n, Nx, Ny, PER, PER);
    dg::DVec w2d = create::w2d( grid);

    dg::DVec y0 = evaluate( sine, grid), y1(y0);

    RHS<dg::DVec> rhs( grid, nu);
    dg::AB< k, dg::DVec > ab( y0);
    dg::Karniadakis< dg::DVec > karn( y0, y0.size(), 1e-6);

    ab.init( rhs, y0, dt);
    Identity<dg::DVec> id(grid);
    karn.init( id, rhs, y0, dt);
    for( unsigned i=0; i<NT; i++)
    {
        //ab( rhs, y0);
        karn( id, rhs, y0);
    }
    DVec solution = evaluate( sol, grid), error( solution);
    double norm_sol = blas2::dot( w2d, solution);
    double norm_y0 = blas2::dot( w2d, y0);
    cout << "Normalized y0 after "<< NT <<" steps is "<< norm_y0 << endl;
    dg::blas1::axpby( -1., y0, 1., error);
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
