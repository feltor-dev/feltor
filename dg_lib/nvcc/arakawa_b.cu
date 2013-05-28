#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "preconditioner2d.cuh"
#include "evaluation.cuh"
#include "arakawa.cuh"
#include "blas.h"
#include "typedefs.cuh"

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
    Grid<double, n> grid( 0, lx, 0, ly, Nx, Ny, dg::PER, dg::PER);
    S2D<double,n > s2d( grid.hx(), grid.hy());
    cout << "# of 2d cells                     " << Nx*Ny <<endl;
    cout << "# of Legendre nodes per dimension "<< n <<endl;
    DVec lhs = expand ( left, grid), jac(lhs);
    DVec rhs = expand ( right,grid);
    const DVec sol = expand( jacobian, grid );
    DVec eins = expand( one, grid );


    Arakawa<double, n, DVec> arakawa( grid);
    t.tic(); 
    arakawa( lhs, rhs, jac);
    t.toc();
    cout << "\nArakawa took "<<t.diff()<<"s\n\n";
    cudaThreadSynchronize();
    //cout<<jac<<endl;


    cout << scientific;
    cout << "Mean     Jacobian is "<<blas2::dot( eins, s2d, jac)<<"\n";
    cout << "Mean rhs*Jacobian is "<<blas2::dot( rhs, s2d, jac)<<"\n";
    cout << "Mean   n*Jacobian is "<<blas2::dot( lhs, s2d, jac)<<"\n";
    blas1::axpby( 1., sol, -1., jac);
    cudaThreadSynchronize();
    cout << "Distance to solution "<<sqrt(blas2::dot( s2d, jac))<<endl; //don't forget sqrt when comuting errors
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
