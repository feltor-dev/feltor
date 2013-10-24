#include <iostream>
#include <fstream>
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

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;


//choose some mean function (attention on lx and ly)
/*
//THESE ARE NOT PERIODIC
double left( double x, double y) { return sin(x/2)*sin(x/2)*exp(x)*sin(y/2.)*sin(y/2.)*log(y+1); }
double right( double x, double y){ return sin(y/2.)*sin(y/2.)*exp(y)*sin(x/2)*sin(x/2)*log(x+1); }
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
const unsigned nmax = 5;
const unsigned Nmin = 4;
const unsigned N = 20;

int main( int argc, char* argv[])
{
    if( argc < 2)
    {
        cerr << "ERROR: Usage arakawa2d_b.cu [outfile.dat]\n";
        return -1;
    }
    std::ofstream ost( argv[1]);
    ost << "# points jacobian lhs*jac rhs*jac error"<<std::endl;
    unsigned n, Nx, Ny;
    for( unsigned i=1; i<=5; i++)
    {
        cout << "P = "<< i <<endl;
        for( unsigned j=0; j<=N; j++)
        {
            n=i;
            Nx = Ny = (unsigned)((double)Nmin*pow( 2., (double)j/4.));
            Grid<double> grid( 0, lx, 0, ly, n, Nx, Ny, dg::PER, dg::PER);
            DVec w2d = create::w2d( grid);
            DVec lhs = evaluate ( left, grid), jac(lhs);
            DVec rhs = evaluate ( right,grid);
            const DVec sol = evaluate( jacobian, grid );
            DVec eins = evaluate( one, grid );
            ArakawaX<DVec> arakawa( grid);
            arakawa( lhs, rhs, jac);

            ost << Nx <<" ";
            ost << fabs(blas2::dot( eins, w2d, jac))<<" ";
            ost << fabs(blas2::dot( lhs, w2d, jac))<<" ";
            ost << fabs(blas2::dot( rhs, w2d, jac))<<" ";
            blas1::axpby( 1., sol, -1., jac);
            ost << sqrt(blas2::dot( w2d, jac))<<endl; //don't forget sqrt when comuting errors
        }
        ost << "\n\n"; //gnuplot next data set
    }
    ost.close();

    return 0;
}
