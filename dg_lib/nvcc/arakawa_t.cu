#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "preconditioner2d.cuh"
#include "evaluation.cuh"
#include "arakawa.cuh"
#include "blas.h"

using namespace std;
using namespace dg;

const unsigned n = 3;
const unsigned Nx = 50;
const unsigned Ny = 50;
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
const double hx = lx/(double)Nx;
const double hy = ly/(double)Ny;

typedef thrust::device_vector<double> DVec;
//typedef thrust::host_vector<double> DVec;
typedef ArrVec2d<double, n, DVec > DArrVec;
typedef cusp::device_memory MemorySpace;

//choose some mean function
double initial( double x, double y) { return sin(x/2)*sin(x/2)*exp(x)*sin(y/2.)*sin(y/2.)*log(y+1); }
double function( double x, double y){ return sin(y/2.)*sin(y/2.)*exp(y)*sin(x/2)*sin(x/2)*log(x+1); }
double one ( double x, double y) {return 1;}



int main()
{
    cout << "# of 2d cells                     " << Nx*Ny <<endl;
    cout << "# of Legendre nodes per dimension "<< n <<endl;
    cout <<fixed<< setprecision(2)<<endl;
    DArrVec init = expand< double(&)(double, double), n> ( initial, 0, lx, 0, ly, Nx, Ny), step(init);
    DArrVec phi = expand< double(&)(double, double), n> ( function, 0, lx, 0, ly, Nx, Ny);
    DArrVec eins = expand< double(&)(double, double), n> ( one, 0, lx, 0, ly, Nx, Ny);
    Arakawa<double, n, DVec, MemorySpace> arakawa( Nx, Ny, hx, hy, init.data());

    arakawa( init.data(), phi.data(), step.data());
    //cout << "Poisson bracket:\n";
    //cout << step << endl;
    cudaThreadSynchronize();
    cout << scientific;
    cout << "Mean     Jacobian is "<<blas2::dot( eins.data(), S2D<double, n>(hx, hy), step.data())<<"\n";
    cout << "Mean phi*Jacobian is "<<blas2::dot( phi.data(), S2D<double, n>(hx, hy), step.data())<<"\n";
    cout << "Mean   n*Jacobian is "<<blas2::dot( init.data(), S2D<double, n>(hx, hy), step.data())<<"\n";

    return 0;
}
