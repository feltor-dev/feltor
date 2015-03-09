#include <iostream>

#include <cusp/ell_matrix.h>

#include "blas.h"
#include "dxx.cuh"
#include "evaluation.cuh"
#include "typedefs.cuh"
#include "weights.cuh"


unsigned n = 3;
unsigned N = 40;
const double lx = 2*M_PI;

/*
double function( double x) { return sin(x);}
double derivative( double x) { return cos(x);}
bc bcx = PER;
double function (double  x) {return x*(x-2*M_PI)*exp(x);}
double derivative( double x) { return (2.*x-2*M_PI)*exp(x) + function(x);}
bc bcx = DIR;
*/
/*
double function( double x) { return cos(x);}
double derivative( double x) { return -sin(x);}
bc bcx = NEU;
*/
double function( double x) { return sin(x);}
double derivative( double x) { return cos(x);}
dg::bc bcx = dg::DIR;
/*
double function( double x) { return cos(3./4.*x);}
double derivative( double x) { return -3./4.*sin(3./4.*x);}
bc bcx = NEU_DIR;
*/

int main ()
{
    std::cout << "Note the supraconvergence!\n";
    std::cout << "Type in n an Nx!\n";
    std::cin >> n>> N;
    std::cout << "# of Legendre nodes " << n <<"\n";
    std::cout << "# of cells          " << N <<"\n";
    dg::Grid1d<double> g( 0, lx, n, N);
    dg::HMatrix hm = dg::create::laplace1d<double>( g, bcx, dg::normed, dg::forward);
    const dg::HVec hv = dg::evaluate( function, g);
    dg::HVec hw = hv;
    //const dg::HVec hu = dg::evaluate( derivative, g);

    dg::blas2::symv( hm, hv, hw);
    dg::blas1::axpby( 1., hv, -1., hw);
    
    std::cout << "Distance to true solution: "<<sqrt(dg::blas2::dot( dg::create::weights(g), hw) )<<"(Note the supraconvergence)\n";
    //for periodic bc | dirichlet bc
    //n = 1 -> p = 2      2
    //n = 2 -> p = 1      1
    //n = 3 -> p = 3      3
    //n = 4 -> p = 3      3
    //n = 5 -> p = 5      5


    
    return 0;
}
