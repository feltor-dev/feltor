#include <iostream>
#include <cusp/print.h>
#include "projection.cuh"
#include "evaluation.cuh"
#include "blas.h"
#include "typedefs.cuh"

double sine( double x){ return sin(x);}
double sine( double x, double y){return sin(x)*sin(y);}

int main()
{
    std::cout << "TEST 1D\n";
    unsigned n_old = 4, n_new = 3, N_old = 10, N_new = 1;
    std::cout << "Type n and N of old (fine) grid!\n";
    std::cin >> n_old >> N_old;
    std::cout << "Type n and N of new (coarser) grid!\n";
    std::cin >> n_new >> N_new;
    dg::Grid1d go ( 0, M_PI, n_old, N_old);
    dg::Grid1d gn ( 0, M_PI, n_new, N_new);
    cusp::coo_matrix<int, double, cusp::host_memory> proj = dg::create::transformation( gn, go);
    cusp::coo_matrix<int, double, cusp::host_memory> inte = dg::create::interpolation( gn, go);
    thrust::host_vector<double> v = dg::evaluate( sine, go);
    thrust::host_vector<double> w1do = dg::create::weights( go);
    thrust::host_vector<double> w1dn = dg::create::weights( gn);
    dg::HVec oneo( go.size(), 1.);
    dg::HVec onen( gn.size(), 1.);
    thrust::host_vector<double> w( gn.size());
    dg::blas2::gemv( proj, v, w);
    std::cout << "Original vector  "<<dg::blas2::dot( oneo, w1do, v) << "\n";
    std::cout << "Projected vector "<<dg::blas2::dot( onen, w1dn, w) << "\n";
    std::cout << "Difference       "<<dg::blas2::dot( oneo, w1do, v) - dg::blas2::dot( onen, w1dn, w) << "\n"<<std::endl;
    dg::blas2::gemv( inte, v, w);
    std::cout << "Original vector  "<<dg::blas2::dot( oneo, w1do, v) << "\n";
    std::cout << "Interpolated vec "<<dg::blas2::dot( onen, w1dn, w) << "\n";
    std::cout << "Difference       "<<dg::blas2::dot( oneo, w1do, v) - dg::blas2::dot( onen, w1dn, w) << "\n"<<std::endl;

    std::cout << "TEST 2D\n";
    
    dg::Grid2d g2o (0, M_PI, 0, M_PI, n_old, N_old, N_old);
    dg::Grid2d g2n (0, M_PI, 0, M_PI, n_new, N_new, N_new);
    cusp::coo_matrix<int, double, cusp::host_memory> proj2d = dg::create::transformation( g2n, g2o);
    cusp::coo_matrix<int, double, cusp::host_memory> inte2d = dg::create::interpolation( g2n, g2o);
    const dg::HVec sinO = dg::evaluate( sine, g2o), 
                   sinN = dg::evaluate( sine, g2n);
    dg::HVec w2do = dg::create::weights( g2o);
    dg::HVec w2dn = dg::create::weights( g2n);
    dg::HVec sinP( g2n.size());
    dg::blas2::gemv( proj2d, sinO, sinP);
    std::cout << "Original vector     "<<sqrt(dg::blas2::dot( sinO, w2do, sinO)) << "\n";
    std::cout << "Projected vector    "<<sqrt(dg::blas2::dot( sinP, w2dn, sinP)) << "\n";
    std::cout << "Difference in Norms "<<sqrt(dg::blas2::dot( sinO, w2do, sinO)) - sqrt(dg::blas2::dot( sinP, w2dn, sinP)) << std::endl;
    std::cout << "Difference between projection and evaluation      ";
    dg::blas1::axpby( 1., sinN, -1., sinP);
    std::cout << sqrt(dg::blas2::dot( sinP, w2dn, sinP)/dg::blas2::dot(sinN, w2dn, sinN))<<"\n";
    dg::blas2::gemv( inte2d, sinO, sinP);
    std::cout << "Interpolated vec    "<<sqrt(dg::blas2::dot( sinP, w2dn, sinP)) << "\n";
    std::cout << "Difference in Norms "<<sqrt(dg::blas2::dot( sinO, w2do, sinO)) - sqrt(dg::blas2::dot( sinP, w2dn, sinP)) << "\n" << std::endl;

    return 0;
}
