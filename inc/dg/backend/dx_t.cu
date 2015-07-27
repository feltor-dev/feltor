#include <iostream>

//#include <cusp/ell_matrix.h>

#include "blas.h"
#include "dx.cuh"
#include "evaluation.cuh"
#include "typedefs.cuh"
#include "weights.cuh"

const double lx = 2*M_PI;
dg::direction dir = dg::centered;

/*
double function( double x) { return sin(x);}
double derivative( double x) { return cos(x);}
bc bcx = PER;
*/

double function (double  x) {return x*(x-2*M_PI)*exp(x);}
double derivative( double x) { return (2.*x-2*M_PI)*exp(x) + function(x);}
dg::bc bcx = dg::DIR;


/*
double function( double x) { return cos(x);}
double derivative( double x) { return -sin(x);}
bc bcx = NEU;
*/

/*
double function( double x) { return sin(3./4.*x);}
double derivative( double x) { return 3./4. * sin(3./4.*x);}
dg::bc bcx = dg::DIR_NEU;
*/

/*
double function( double x) { return cos(3./4.*x);}
double derivative( double x) { return -3./4.*sin(3./4.*x);}
bc bcx = NEU_DIR;
*/

int main ()
{
    unsigned int n = 3;
    unsigned int N = 40;

    std::cout << "Note the supraconvergence!\n";
    std::cout << "Type in n an Nx!\n";
    std::cin >> n>> N;
    std::cout << "# of Legendre nodes " << n <<"\n";
    std::cout << "# of cells          " << N <<"\n";
    dg::Grid1d<double> g( 0, lx, n, N);
    const double hx = lx/(double)N;
  cusp::ell_matrix< int, double, cusp::host_memory> hm = dg::create::dx_symm_normed<double>( n, N, hx, bcx);
//  cusp::ell_matrix< int, double, cusp::host_memory> hm = dg::create::dx_minus_normed<double>( n, N, hx, bcx);
//  cusp::ell_matrix< int, double, cusp::host_memory> hm = dg::create::dx_plus_normed<double>( n, N, hx, bcx);
    dg::HVec hv = dg::evaluate( function, g);
    dg::HVec hw = hv;
    const dg::HVec hu = dg::evaluate( derivative, g);


//    std::cout << "Input vector: " << std::endl;
//    for(dg::HVec::iterator it = hv.begin(); it != hv.end(); it++)
//    {
//        std::cout << *it << "  ";
//    }
//    std::cout << std::endl;

    dg::blas2::symv( hm, hv, hw);

//    std::cout << "True solution:" << std::endl;
//    for(unsigned int i = 0; i < hu.size(); i++)
//    {
//        std::cout << hu[i] << " ";
//    }
//    std::cout << std::endl;

//    std::cout << "Our solution:" << std::endl;
//    for(unsigned int i = 0; i < hu.size(); i++)
//    {
//        std::cout << hw[i] << " ";
//    }
//    std::cout << std::endl;


    dg::blas1::axpby( 1., hu, -1., hw);
    //
    std::cout << "Distance to true solution (cusp_matrix): "<<sqrt(dg::blas2::dot( dg::create::weights(g), hw) )<<"\n";

    dg::dx_matrix hm_2(n, N, hx, bcx, dir);
    hv = dg::evaluate(function, g);
    hw = hv;
    dg::blas2::symv(hm_2, hv, hw);
    dg::blas1::axpby(1., hu, -1., hw);

    std::cout << "Distance to true solution (dx_matrix): "<<sqrt(dg::blas2::dot( dg::create::weights(g), hw) )<<"\n";

    //for periodic bc | dirichlet bc
    //n = 1 -> p = 2      2
    //n = 2 -> p = 1      1
    //n = 3 -> p = 3      3
    //n = 4 -> p = 3      3
    //n = 5 -> p = 5      5


    
    return 0;
}
