#include <iostream>
#include <fstream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "backend/evaluation.cuh"
#include "poisson.h"
#include "blas.h"
#include "backend/typedefs.cuh"

#include "backend/timer.cuh"

using namespace std;
using namespace dg;

//const double lx = M_PI;
//const double ly = M_PI;
const double eps = 1e-1;


//choose some mean function (attention on lx and ly)
//THESE ARE NOT PERIODIC
/*
double left( double x, double y) { return sin(x/2)*sin(x/2)*exp(x)*sin(y/2.)*sin(y/2.)*log(y+1); }
double right( double x, double y){ return sin(y/2.)*sin(y/2.)*exp(y)*sin(x/2)*sin(x/2)*log(x+1); }
*/
//double left( double x, double y) {return sin(x)*exp(x-M_PI)*sin(y);}
//double right( double x, double y) {return sin(x)*sin(y)*exp(y-M_PI);}
//double jacobian( double x, double y) 
//{
//    return exp( x-M_PI)*(sin(x)+cos(x))*sin(y) * exp(y-M_PI)*sin(x)*(sin(y) + cos(y)) - sin(x)*exp(x-M_PI)*cos(y) * cos(x)*sin(y)*exp(y-M_PI); 
//}
//schlechte Funktion für Conservation benchmarks
double left( double x, double y) {
    return sin(0.5*M_PI*x*x)*sin(0.5*M_PI*y);
}
double right( double x, double y) {
    return sin( M_PI*x*x*x)*sin(M_PI*y*y)/(2.+sin(M_PI*x)-0.5);
}
double jacobian( double x, double y) 
{
    //return cos(x )*cos(y )*cos(x )*cos(y ) - sin(x)*sin(y)*sin(x)*sin(y); 
    return cos(x )*cos(y )*exp(x+y) - sin(x)*sin(y)*exp(x+y); 
}
////These are for comparing to FD arakawa results
//double left( double x, double y) {return sin(2.*M_PI*(x-hx/2.));}
//double right( double x, double y) {return y;}
//double jacobian( double x, double y) {return 2.*M_PI*cos(2.*M_PI*(x-hx/2.));}
const unsigned nmax = 4;
const unsigned Nmin = 7;
const unsigned N =3; 

int main( int argc, char* argv[])
{
    //if( argc < 2)
    //{
    //    cerr << "ERROR: Usage arakawa2d_b.cu [outfile.dat]\n";
    //    return -1;
    //}
    //std::ofstream ost( argv[1]);
    //ost << "# points jacobian lhs*jac rhs*jac error"<<std::endl;
    std::cout << "# points jacobian lhs*jac rhs*jac error"<<std::endl;
    unsigned n, Nx, Ny;
    for( unsigned i=2; i<=nmax; i++)
    {
        cout << "P = "<< i <<endl;
        for( unsigned j=0; j<=N; j++)
        {
            n=i;
            Nx = Ny = (unsigned)((double)Nmin*pow( 2., (double)j));
            Grid2d<double> grid( -2., 2., -2., 2., n, Nx, Ny, dg::PER, dg::PER);
            DVec w2d = create::weights( grid);
            DVec lhs = evaluate ( left, grid), jac(lhs);
            DVec rhs = evaluate ( right,grid);
            const DVec sol = evaluate( jacobian, grid );
            DVec eins = evaluate( one, grid );
            Poisson<DMatrix, DVec> poiss( grid);
            poiss( lhs, rhs, jac);

            cout << Nx <<" ";
            cout << fabs(blas2::dot( eins, w2d, jac))<<" ";
            cout << fabs(blas2::dot( lhs, w2d, jac))<<" ";
            cout << fabs(blas2::dot( rhs, w2d, jac))<<" ";
            blas1::axpby( 1., sol, -1., jac);
            cout << sqrt(blas2::dot( w2d, jac))<<endl; //don't forget sqrt when comuting errors
        }
        cout << "\n\n"; //gnuplot next data set
    }
    //ost.close();

    return 0;
}
