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

const unsigned n = 1;
const unsigned Nx = 10;
const unsigned Ny = 10;
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
const double hx = lx/(double)Nx;
const double hy = ly/(double)Ny;

typedef thrust::device_vector<double> DVec;
//typedef thrust::host_vector<double> DVec;
typedef ArrVec2d<double, n, DVec > DArrVec;
typedef cusp::device_memory MemorySpace;

//choose some mean function
/*
//THESE ARE NOT PERIODIC AND THUS WON'T CONVERGE TO TRUE SOLUTION
double left( double x, double y) { return sin(x/2)*sin(x/2)*exp(x)*sin(y/2.)*sin(y/2.)*log(y+1); }
double right( double x, double y){ return sin(y/2.)*sin(y/2.)*exp(y)*sin(x/2)*sin(x/2)*log(x+1); }
*/


double left( double x, double y) {return sin(x)*exp(x+2)*sin(y);}
double right( double x, double y) {return sin(x)*sin(y)*exp(y+1);}
double jacobian( double x, double y) 
{
    return exp( x+2)*(sin(x)+cos(x))*sin(y)*exp(y+1)*sin(x)*(sin(y) + cos(y)) - sin(x)*exp(x+2)*cos(y)*cos(x)*sin(y)*exp(y+1); 
}

double one ( double x, double y) {return 1;}

int main()
{
    cout << "# of 2d cells                     " << Nx*Ny <<endl;
    cout << "# of Legendre nodes per dimension "<< n <<endl;
    cout <<fixed<< setprecision(2)<<endl;
    DArrVec lhs = expand< double(&)(double, double), n> ( left, 0, lx, 0, ly, Nx, Ny), jac(lhs);
    DArrVec rhs = expand< double(&)(double, double), n> ( right, 0, lx, 0, ly, Nx, Ny);
    const DArrVec sol = expand< double(&)(double, double), n> ( jacobian, 0, lx, 0, ly, Nx, Ny);
    DArrVec eins = expand< double(&)(double, double), n> ( one, 0, lx, 0, ly, Nx, Ny);
    Arakawa<double, n, DVec, MemorySpace> arakawa( Nx, Ny, hx, hy, lhs.data());

    arakawa( lhs.data(), rhs.data(), jac.data());
    cudaThreadSynchronize();
    cout << scientific;
    cout << "Mean     Jacobian is "<<blas2::dot( eins.data(), S2D<double, n>(hx, hy), jac.data())<<"\n";
    cout << "Mean rhs*Jacobian is "<<blas2::dot( rhs.data(), S2D<double, n>(hx, hy), jac.data())<<"\n";
    cout << "Mean   n*Jacobian is "<<blas2::dot( lhs.data(), S2D<double, n>(hx, hy), jac.data())<<"\n";
    blas1::axpby( 1., sol.data(), -1., jac.data());
    cudaThreadSynchronize();
    cout << "Distance to solution "<<blas2::dot( S2D<double, n>(hx, hy), jac.data())<<endl;

    return 0;
}
