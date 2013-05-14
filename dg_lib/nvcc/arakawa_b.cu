#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "preconditioner2d.cuh"
#include "evaluation.cuh"
#include "arakawa.cuh"
#include "blas.h"

#include "timer.cuh"

using namespace std;
using namespace dg;

const unsigned n = 3;
const unsigned Nx = 100;
const unsigned Ny = 100;
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
//const double lx = 1.;
//const double ly = 1.;
const double hx = lx/(double)Nx;
const double hy = ly/(double)Ny;

typedef thrust::device_vector<double> DVec;
//typedef thrust::host_vector<double> DVec;
typedef ArrVec2d<double, n, DVec > DArrVec;
typedef cusp::device_memory MemorySpace;

//choose some mean function (attention on lx and ly)
/*
//THESE ARE NOT PERIODIC
double left( double x, double y) { return sin(x/2)*sin(x/2)*exp(x)*sin(y/2.)*sin(y/2.)*log(y+1); }
double right( double x, double y){ return sin(y/2.)*sin(y/2.)*exp(y)*sin(x/2)*sin(x/2)*log(x+1); }
*/
/*
double left( double x, double y) {return sin(x)*exp(x-M_PI)*sin(y);}
double right( double x, double y) {return sin(x)*sin(y)*exp(y-M_PI);}
double jacobian( double x, double y) 
{
    return exp( x-M_PI)*(sin(x)+cos(x))*sin(y) * exp(y-M_PI)*sin(x)*(sin(y) + cos(y)) - sin(x)*exp(x-M_PI)*cos(y) * cos(x)*sin(y)*exp(y-M_PI); 
}
*/

double left( double x, double y) {return sin(x)*cos(y);}
double right( double x, double y) {return cos(x)*sin(y);}
double jacobian( double x, double y) 
{
    return cos(x)*cos(y)*cos(x)*cos(y) - sin(x)*sin(y)*sin(x)*sin(y); 
}
////These are for comparing to FD arakawa results
//double left( double x, double y) {return sin(2.*M_PI*(x-hx/2.));}
//double right( double x, double y) {return y;}
//double jacobian( double x, double y) {return 2.*M_PI*cos(2.*M_PI*(x-hx/2.));}

int main()
{
    Timer t;
    cout << "# of 2d cells                     " << Nx*Ny <<endl;
    cout << "# of Legendre nodes per dimension "<< n <<endl;
    DArrVec lhs = expand< double(&)(double, double), n> ( left, 0, lx, 0, ly, Nx, Ny), jac(lhs);
    DArrVec rhs = expand< double(&)(double, double), n> ( right, 0, lx, 0, ly, Nx, Ny);
    const DArrVec sol = expand< double(&)(double, double), n> ( jacobian, 0, lx, 0, ly, Nx, Ny);
    DArrVec eins = expand< double(&)(double, double), n> ( one, 0, lx, 0, ly, Nx, Ny);


    Arakawa<double, n, DVec, MemorySpace> arakawa( Nx, Ny, hx, hy, -1, -1);
    t.tic(); 
    arakawa( lhs.data(), rhs.data(), jac.data());
    t.toc();
    cout << "\nArakawa took "<<t.diff()<<"s\n\n";
    cudaThreadSynchronize();
    //cout<<jac<<endl;


    cout << scientific;
    cout << "Mean     Jacobian is "<<blas2::dot( eins.data(), S2D<double, n>(hx, hy), jac.data())<<"\n";
    cout << "Mean rhs*Jacobian is "<<blas2::dot( rhs.data(), S2D<double, n>(hx, hy), jac.data())<<"\n";
    cout << "Mean   n*Jacobian is "<<blas2::dot( lhs.data(), S2D<double, n>(hx, hy), jac.data())<<"\n";
    blas1::axpby( 1., sol.data(), -1., jac.data());
    cudaThreadSynchronize();
    cout << "Distance to solution "<<sqrt(blas2::dot( S2D<double, n>(hx, hy), jac.data()))<<endl; //don't forget sqrt when comuting errors
    //periocid bc       |  dirichlet bc
    //n = 1 -> p = 2    |     
    //n = 2 -> p = 1    |
    //n = 3 -> p = 3    |        3
    //n = 4 -> p = 3    | 
    //n = 5 -> p = 5    |
    // quantities are all conserved to 1e-15 for periodic bc
    // for dirichlet bc these are not better conserved than normal jacobian

    return 0;
}
