#include <iostream>

#include <cusp/ell_matrix.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "rk.cuh"
#include "arrvec1d.cuh"
#include "evaluation.cuh"
#include "laplace.cuh"
#include "preconditioner.cuh"

#include "blas.h"

template <class T, size_t n, class container = thrust::device_vector<T>, class MemorySpace = cusp::device_memory>
struct RHS
{
    typedef container Vector;
    RHS(unsigned N, T h, T D):h(h), D(D) 
    {
        laplace = dg::create::laplace1d_dir<T,n>( N, h, true);
    }
    void operator()( const container& y, container& yp)
    {
        dg::blas2::symv( laplace, y, yp);
        dg::blas1::axpby( -D, yp, 0., yp);
        //laplace is unnormalized -laplace
        //dg::blas2::symv( -D, dg::T1D<T,n>(h), yp, 0., yp); 
    }
  private:
    double h, D;
    cusp::ell_matrix<int, T, MemorySpace> laplace;
};

const unsigned n = 3;
const unsigned N = 80;
const double lx = 2.*M_PI;

const unsigned k = 3;
const double nu = 0.01;
const double T = 0.8;
const unsigned NT = (unsigned)(nu*T*n*n*N*N/0.01/lx/lx);

double sine( double x) {return sin(x);}
double sol( double x) {return exp( -nu*T)*sine(x);}



//typedef thrust::device_vector<double> DVec;
typedef thrust::host_vector<double> DVec;
typedef dg::ArrVec1d<double, n, DVec> DArrVec;
typedef cusp::host_memory MemorySpace;
    

using namespace std;
using namespace dg;

int main()
{
    const double dt =T/(double)NT;
    cout << "Test RK scheme on diffusion equation\n";
    cout << "Polynomial coefficients:  "<< n<<endl;
    cout << "RK order K:               "<< k <<endl;
    cout << "Number of gridpoints:     "<<N<<endl;

    Grid1d<double, n> grid( 0, lx, N);
    S1D<double, n> s1d( grid.h());

    DVec y0 = expand( sine, grid), y1(y0);
    double norm_y0 = dg::blas2::dot( s1d, y0);
    cout << "Normalized y0 (is Pi) "<< norm_y0 << endl;

    RHS<double,n, DVec, MemorySpace> rhs( grid.N(), grid.h(), nu);
    RK< k, RHS<double, n, DVec, MemorySpace> > rk( y0);
    AB< k, RHS<double, n, DVec, MemorySpace> > ab( y0);

    ab.init( rhs, y0, dt);
    //thrust::swap(y0, y1);
    for( unsigned i=0; i<NT; i++)
    {
        ab( rhs, y0, y1, dt);
        thrust::swap(y0, y1);
    }
    norm_y0 = blas2::dot( s1d, y0);
    cout << "Normalized y0 after "<< NT <<" steps is "<< norm_y0 << endl;
    DVec solution = expand( sol, grid), error( solution);
    double norm_sol = blas2::dot( s1d, solution);
    blas1::axpby( -1., y0, 1., error);
    cout << "Normalized solution is "<<  norm_sol<< endl;
    double norm_error = blas2::dot( s1d, error);
    cout << "Relative error is      "<< sqrt( norm_error/norm_sol)<< endl;
    //n = 1 -> p = 1 (Sprung in laplace macht n=1 eine Ordng schlechter) 
    //n = 2 -> p = 2
    //n = 3 -> p = 3
    //n = 4 -> p = 4
    //n = 5 -> p = 5

    return 0;
}
