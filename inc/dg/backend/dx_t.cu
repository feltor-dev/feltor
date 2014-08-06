#include <iostream>

#include <cusp/ell_matrix.h>

#include "blas.h"
#include "dx.cuh"
#include "evaluation.cuh"
#include "preconditioner.cuh"
#include "typedefs.cuh"


using namespace std;
using namespace dg;

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
bc bcx = DIR_NEU;
/*
double function( double x) { return cos(3./4.*x);}
double derivative( double x) { return -3./4.*sin(3./4.*x);}
bc bcx = NEU_DIR;
*/

int main ()
{
    cout << "Note the supraconvergence!\n";
    cout << "Type in n an Nx!\n";
    cin >> n>> N;
    cout << "# of Legendre nodes " << n <<"\n";
    cout << "# of cells          " << N <<"\n";
    Grid1d<double> g( 0, lx, n, N);
    const double hx = lx/(double)N;
    //cusp::ell_matrix< int, double, cusp::host_memory> hm = create::dx_symm<double>( n, N, hx, bcx);
    cusp::ell_matrix< int, double, cusp::host_memory> hm = create::dx_minus_mt<double>( n, N, hx, bcx);
    HVec hv = expand( function, g);
    HVec hw = hv;
    const HVec hu = expand( derivative, g);

    blas2::symv( hm, hv, hw);
    blas1::axpby( 1., hu, -1., hw);
    
    cout << "Distance to true solution: "<<sqrt(blas2::dot( S1D<double>(g), hw) )<<"\n";
    //for periodic bc | dirichlet bc
    //n = 1 -> p = 2      2
    //n = 2 -> p = 1      1
    //n = 3 -> p = 3      3
    //n = 4 -> p = 3      3
    //n = 5 -> p = 5      5


    
    return 0;
}
