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
const unsigned Nx = 50;
const unsigned Ny = 50;
//const double lx = 2.*M_PI;
//const double ly = 2.*M_PI;



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

/*
double left( double x, double y) {return sin(x)*cos(y);}
double right( double x, double y) {return sin(y)*cos(x);} 
dg::bc bcx = dg::PER; 
dg::bc bcy = dg::PER;
//double right2( double x, double y) {return sin(y);}
double jacobian( double x, double y) 
{
    return cos(x)*cos(y)*cos(x)*cos(y) - sin(x)*sin(y)*sin(x)*sin(y); 
}
*/
////These are for comparing to FD arakawa results
//double left( double x, double y) {return sin(2.*M_PI*(x-hx/2.));}
//double right( double x, double y) {return y;}
//double jacobian( double x, double y) {return 2.*M_PI*cos(2.*M_PI*(x-hx/2.));}
const double lx = M_PI/2.;
const double ly = M_PI/2.;
double left( double x, double y) {return sin(x)*sin(y);}
double right( double x, double y) {return sin(2*x)*sin(2*y);} 
dg::bc bcx = dg::DIR_NEU; 
dg::bc bcy = dg::DIR_NEU;
//double right2( double x, double y) {return sin(y);}
double jacobian( double x, double y) 
{
    return cos(x)*sin(y)*2*sin(2*x)*cos(2*y)-sin(x)*cos(y)*2*cos(2*x)*sin(2*y);
}

const unsigned Nz = 10;
const double lz = 1.;
double left( double x, double y, double z) {return left(x,y)*z;}
double right( double x, double y, double z) {return right(x,y)*z;} 
//double right2( double x, double y) {return sin(y);}
double jacobian( double x, double y, double z) 
{
    return jacobian(x,y)*z*z;
}

int main()
{
    std::cout << "TEST 3d derivatives!\n";
    Grid3d<double> g3d( 0, lx, 0, ly, 0, lz, n, Nx, Ny, Nz, bcx, bcy);
    DVec w3d = create::w3d( g3d);
    cout << "# of 2d cells                     " << Nx*Ny <<endl;
    cout << "# of Legendre nodes per dimension "<< n <<endl;
    cout <<fixed<< setprecision(2)<<endl;
    DVec lhs3 = evaluate( left, g3d), jac3(lhs3);
    DVec rhs3 = evaluate( right, g3d);
    const DVec sol3 = evaluate( jacobian, g3d);
    DVec eins3 = evaluate( one, g3d);

    ArakawaX< DVec> arakawa3( g3d);
    arakawa3( lhs3, rhs3, jac3);

    cout << scientific;
    cout << "Mean     Jacobian is "<<blas2::dot( eins3, w3d, jac3)<<"\n";
    cout << "Mean rhs*Jacobian is "<<blas2::dot( rhs3,  w3d, jac3)<<"\n";
    cout << "Mean lhs*Jacobian is "<<blas2::dot( lhs3,  w3d, jac3)<<"\n";
    blas1::axpby( 1., sol3, -1., jac3);
    cout << "Distance to solution "<<sqrt( blas2::dot( w3d, jac3))<<endl; //don't forget sqrt when comuting errors

    return 0;
}

