#include <iostream>

#include <cusp/ell_matrix.h>

#include "blas.h"
#include "dx.h"
#include "dx.cuh"
#include "sparseblockmat.cuh"
#include "evaluation.cuh"
#include "typedefs.cuh"
#include "weights.cuh"


unsigned n = 3;
unsigned N = 40;
const double lx = 2*M_PI;

//double function( double x) { return sin(x);}
//double derivative( double x) { return cos(x);}
////dg::bc bcx = dg::PER;
//dg::bc bcx = dg::DIR;

//double function (double  x) {return x*(x-2*M_PI)*exp(x);}
//double derivative( double x) { return (2.*x-2*M_PI)*exp(x) + function(x);}
//dg::bc bcx = dg::DIR;
/*
double function( double x) { return cos(x);}
double derivative( double x) { return -sin(x);}
dg::bc bcx = dg::NEU;
*/
double function( double x) { return sin(3./4.*x);}
double derivative( double x) { return 3./4.*cos(3./4.*x);}
dg::bc bcx = dg::DIR_NEU;
/*
double function( double x) { return cos(3./4.*x);}
double derivative( double x) { return -3./4.*sin(3./4.*x);}
dg::bc bcx = NEU_DIR;
*/

typedef dg::DVec Vector;
typedef dg::SparseBlockMatGPU Matrix;

int main ()
{
    std::cout << "Type in n an Nx!\n";
    std::cin >> n>> N;
    std::cout << "# of Legendre nodes " << n <<"\n";
    std::cout << "# of cells          " << N <<"\n";
    dg::Grid1d<double> g( 0, lx, n, N);
    const double hx = lx/(double)N;

    Matrix hs = dg::create::dx_symm( n, N, hx, bcx);
    Matrix hf = dg::create::dx_plus( n, N, hx, bcx);
    Matrix hb = dg::create::dx_minus( n, N, hx, bcx);
    const Vector hv = dg::evaluate( function, g);
    Vector hw = hv;
    Vector w1d = dg::create::weights( g);
    const Vector hu = dg::evaluate( derivative, g);

    dg::blas2::symv( hs, hv, hw);
    dg::blas1::axpby( 1., hu, -1., hw);
    std::cout << "Distance to true solution (symmetric): "<<sqrt(dg::blas2::dot( w1d, hw) )<<"\n";
    dg::blas2::symv( hf, hv, hw);
    dg::blas1::axpby( 1., hu, -1., hw);
    std::cout << "Distance to true solution (forward  ): "<<sqrt(dg::blas2::dot( w1d, hw) )<<"\n";
    dg::blas2::symv( hb, hv, hw);
    dg::blas1::axpby( 1., hu, -1., hw);
    std::cout << "Distance to true solution (backward ): "<<sqrt(dg::blas2::dot( w1d, hw) )<<"\n";
    //for periodic bc | dirichlet bc
    //n = 1 -> p = 2      2
    //n = 2 -> p = 1      1
    //n = 3 -> p = 3      3
    //n = 4 -> p = 3      3
    //n = 5 -> p = 5      5


    
    return 0;
}
