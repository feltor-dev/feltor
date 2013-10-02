#include <iostream>
#include "projection.cuh"
#include "evaluation.cuh"
#include "blas.h"

double sine( double x){ return sin(x);}

int main()
{
    unsigned n_old = 3, n_new = 3, N = 10, Nf = 4;
    dg::Grid1d<double> g  ( 0, 2.*M_PI, n_old, N);
    dg::Grid1d<double> gn ( 0, 2.*M_PI, n_new, N*Nf);
    std::vector<double> p = dg::create::projection( n_old, n_new, Nf);
    cusp::coo_matrix<int, double, cusp::host_memory> proj = dg::create::diagonal_matrix( N, p, n_new*Nf, n_old);
    
    thrust::host_vector<double> v = dg::evaluate( sine, g);
    thrust::host_vector<double> w1d = dg::create::w1d( g);
    thrust::host_vector<double> w1dn = dg::create::w1d( gn);
    thrust::host_vector<double> w( n_new*Nf*N);
    dg::blas2::gemv( proj, v, w);
    std::cout << dg::blas2::dot( v, w1d, v) << "\n";
    std::cout << dg::blas2::dot( w, w1dn, w) << "\n";

    return 0;
}
