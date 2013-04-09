#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "preconditioner2d.cuh"
#include "evaluation.cuh"
#include "arakawa.cuh"
#include "rk.cuh"

using namespace std;
using namespace dg;

const unsigned n = 3;
const unsigned Nx = 10;
const unsigned Ny = 10;
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;

const unsigned k = 3;
const double T = 1.;
const double NT = 10;

typedef thrust::device_vector<double> DVec;
typedef ArrVec2d<double, n, DVec > DArrVec;

double initial( double x, double y)
{
    return sin(x)*sin(y);
}
double function( double x, double y)
{
    return sin(y);
}
double result( double x, double y)
{
    return initial( x-cos(y)*T, y);
}

const double hx = lx/(double)Nx;
const double hy = ly/(double)Ny;
const double dt = T/(double)NT;

struct RHS
{
    typedef thrust::device_vector<double> Vector;
    RHS(): arakawa( Nx, Ny, hx, hy), phi( expand<double(&)(double, double), n>( function, 0, lx, 0, ly, Nx, Ny)){}
    void operator()( const DVec& y, DVec& yp)
    {
        cout << "Hello\n";
        arakawa( phi.data(), y, yp);
        cout << "Bye\n";
    }
  private:
    Arakawa<double, n> arakawa;
    DArrVec phi;
};

int main()
{
    DArrVec init = expand< double(&)(double, double), n> ( initial, 0, lx, 0, ly, Nx, Ny), step;
    DArrVec res = expand< double(&)(double, double), n> ( result, 0, lx, 0, ly, Nx, Ny);

    RHS rhs;
    cout << "ping after\n";
    RK<3, RHS>  rk( init.data());
    cout << "ping after2\n";
    for( unsigned i=0; i<NT/2; i++)
    {
        rk( rhs, init.data(), step.data(), dt);
        init = step;
    }

    cout << "Hello world\n";
    blas1::axpby( 1., res.data(), -1., step.data());
    cudaThreadSynchronize();
    cout << "Norm of error is "<<blas2::dot( T2D<double, n>(hx, hy), step.data())<<"\n";;

    return 0;
}
