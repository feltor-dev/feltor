#include <iostream>
#include "projection.cuh"
#include "evaluation.cuh"
#include "blas.h"

double sine( double x){ return sin(x);}
double sine( double x, double y){return sin(x)*sin(y);}

int main()
{
    std::cout << "TEST 1D\n";
    unsigned n_old = 4, n_new = 3, N = 10, Nf = 4;
    dg::Grid1d<double> g  ( 0, 2.*M_PI, n_old, N);
    dg::Grid1d<double> gn ( 0, 2.*M_PI, n_new, N*Nf);
    std::cout << "Create Projection\n";
    cusp::coo_matrix<int, double, cusp::host_memory> proj = dg::create::projection1d( g, gn);
    
    std::cout << "Created Projection\n";
    thrust::host_vector<double> v = dg::evaluate( sine, g);
    thrust::host_vector<double> w1d = dg::create::w1d( g);
    thrust::host_vector<double> w1dn = dg::create::w1d( gn);
    thrust::host_vector<double> w( gn.size());
    std::cout << "Projection:\n";
    dg::blas2::gemv( proj, v, w);
    std::cout << "Original vector  "<<dg::blas2::dot( v, w1d, v) << "\n";
    std::cout << "Projected vector "<<dg::blas2::dot( w, w1dn, w) << "\n";
    std::cout << "Difference       "<<dg::blas2::dot( v, w1d, v) - dg::blas2::dot( w, w1dn, w) << "\n";

    std::cout << "TEST 2D\n";
    dg::Grid<double> g2 (0, 2.*M_PI, 0, 2.*M_PI, n_old, N, N);
    dg::Grid<double> g2n (0, 2.*M_PI, 0, 2.*M_PI, n_new, N, N);
    return 0;
}
