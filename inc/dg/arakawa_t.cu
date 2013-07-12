#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "preconditioner2d.cuh"
#include "evaluation.cuh"
#include "arakawa.cuh"
#include "blas.h"
#include "typedefs.cuh"

using namespace std;
using namespace dg;

const unsigned n = 3;
const unsigned Nx = 25;
const unsigned Ny = 25;
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;


//choose some mean function (attention on lx and ly)
/*
//THESE ARE NOT PERIODIC
double left( double x, double y) { return sin(x/2)*sin(x/2)*exp(x)*sin(y/2.)*sin(y/2.)*log(y+1); }
double right( double x, double y){ return sin(y/2.)*sin(y/2.)*exp(y)*sin(x/2)*sin(x/2)*log(x+1); }
*/
/*
double left( double x, double y) {return sin(x)*exp(x-M_PI)*sin(y);}
double right( double x, double y) {return sin(x)*sin(y);}
double right2( double x, double y) {return exp(y-M_PI);}
double jacobian( double x, double y) 
{
    return exp( x-M_PI)*(sin(x)+cos(x))*sin(y) * exp(y-M_PI)*sin(x)*(sin(y) + cos(y)) - sin(x)*exp(x-M_PI)*cos(y) * cos(x)*sin(y)*exp(y-M_PI); 
}
*/

double left( double x, double y) {return sin(x)*cos(y);}
double right( double x, double y) {return cos(x);}
double right2( double x, double y) {return sin(y);}
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
    Grid<double> grid( 0, lx, 0, ly, n, Nx, Ny, dg::PER, dg::PER);
    DVec w2d = create::s2d( grid);
    cout << "# of 2d cells                     " << Nx*Ny <<endl;
    cout << "# of Legendre nodes per dimension "<< n <<endl;
    cout <<fixed<< setprecision(2)<<endl;
    DVec lhs = expand( left, grid), jac(lhs), jac1( lhs), jac2(lhs);
    DVec rhs1 = expand( right, grid), rhs( rhs1);
    DVec rhs2 = expand( right2, grid);
    blas1::pointwiseDot( rhs1, rhs2, rhs);
    const DVec sol = expand ( jacobian, grid);
    DVec eins = expand( one, grid);

    Arakawa< DVec> arakawa( grid);
    arakawa( lhs, rhs, jac);

    arakawa( lhs, rhs1, jac1);
    blas1::pointwiseDot( rhs2, jac1, jac1);
    arakawa( lhs, rhs2, jac2);
    blas1::pointwiseDot( rhs1, jac2, jac2);
    blas1::axpby( 1., jac1, 1., jac2, jac2);

    cout << scientific;
    cout << "Mean     Jacobian is "<<blas2::dot( eins, w2d, jac)<<"\n";
    cout << "Mean     Jacobian is "<<blas2::dot( eins, w2d, jac2)<<"\n";
    cout << "Mean rhs*Jacobian is "<<blas2::dot( rhs,  w2d, jac)<<"\n";
    cout << "Mean rhs*Jacobian is "<<blas2::dot( rhs,  w2d, jac2)<<"\n";
    cout << "Mean lhs*Jacobian is "<<blas2::dot( lhs,  w2d, jac)<<"\n";
    cout << "Mean lhs*Jacobian is "<<blas2::dot( lhs,  w2d, jac2)<<"\n";
    blas1::axpby( 1., sol, -1., jac);
    cout << "Distance to solution "<<sqrt( blas2::dot( w2d, jac))<<endl; //don't forget sqrt when comuting errors
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

