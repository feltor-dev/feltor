#include <iostream>

#include <cusp/ell_matrix.h>

#include "blas.h"
#include "dx.cuh"
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
double function( double x) { return sin(3./4.*x);}
double derivative( double x) { return 3./4.*cos(3./4.*x);}
dg::bc bcx = dg::DIR_NEU;
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
    const double hx = lx/(double)N;
    //cusp::ell_matrix< int, double, cusp::host_memory> hm = dg::create::dx_symm_normed<double>( n, N, hx, bcx);
    //cusp::ell_matrix< int, double, cusp::host_memory> hm = dg::create::dx_minus_normed<double>( n, N, hx, bcx);
    cusp::ell_matrix< int, double, cusp::host_memory> hm = dg::create::dx_plus_normed<double>( n, N, hx, bcx);
    dg::HVec hv = dg::evaluate( function, g);
    dg::HVec hw = hv;
    const dg::HVec hu = dg::evaluate( derivative, g);

    dg::blas2::symv( hm, hv, hw);
    dg::blas1::axpby( 1., hu, -1., hw);
    
    std::cout << "Distance to true solution: "<<sqrt(dg::blas2::dot( dg::create::weights(g), hw) )<<"\n";
    //for periodic bc | dirichlet bc
    //n = 1 -> p = 2      2
    //n = 2 -> p = 1      1
    //n = 3 -> p = 3      3
    //n = 4 -> p = 3      3
    //n = 5 -> p = 5      5


    
    return 0;
}
