#include <iostream>
#include <cusp/print.h>
#include "typedefs.cuh"
#include "projection.cuh"
#include "evaluation.cuh"
#include "blas.h"

double sine( double x){ return sin(x);}
double sine( double x, double y){return sin(x)*sin(y);}

int main()
{
    std::cout << "TEST 1D\n";
    unsigned n_old = 3, n_new = 3, N = 100, Nf = 2;
    dg::Grid1d<double> g  ( 0, 2.*M_PI, n_old, N);
    dg::Grid1d<double> gn ( 0, 2.*M_PI, n_new, N*Nf);
    cusp::coo_matrix<int, double, cusp::host_memory> proj = dg::create::projection1d( g, gn);
    thrust::host_vector<double> v = dg::evaluate( sine, g);
    thrust::host_vector<double> w1d = dg::create::w1d( g);
    thrust::host_vector<double> w1dn = dg::create::w1d( gn);
    thrust::host_vector<double> w( gn.size());
    dg::blas2::gemv( proj, v, w);
    std::cout << "Original vector  "<<dg::blas2::dot( v, w1d, v) << "\n";
    std::cout << "Projected vector "<<dg::blas2::dot( w, w1dn, w) << "\n";
    std::cout << "Difference       "<<dg::blas2::dot( v, w1d, v) - dg::blas2::dot( w, w1dn, w) << "\n"<<std::endl;

    std::cout << "TEST KRONECKER PRODUCT\n";
    dg::create::detail::HelperMatrix<double> m1(2,2), m2(2,2);
    for( unsigned i=0; i<2;i++)
        for( unsigned j=0; j<2;j++)
            m1(i,j) = 2*i+j+1;
    m2(0,0) = 0; m2( 0, 1) = 5; m2( 1,0) = 6, m2( 1,1) = 7;
    std::cout << "M1 \n"<<m1 << "Times\n"<<m2;
    std::cout << "Is \n"<<dg::create::detail::kronecker( m1, m2);
    std::cout << "(Compare Wikipedia for correctness!)\n"<<std::endl;

    std::cout << "TEST 2D\n";
    dg::Grid<double> g2 (0, 2.*M_PI, 0, 2.*M_PI, n_old, N, N);
    dg::Grid<double> g2n (0, 2.*M_PI, 0, 2.*M_PI, n_new, N*Nf, N*Nf);
    cusp::coo_matrix<int, double, cusp::host_memory> proj2d = dg::create::projection2d( g2, g2n);
    thrust::host_vector<double> v2 = dg::evaluate( sine, g2);
    thrust::host_vector<double> w2d = dg::create::w2d( g2);
    thrust::host_vector<double> w2dn = dg::create::w2d( g2n);
    thrust::host_vector<double> w2( g2n.size());
    dg::blas2::gemv( proj2d, v2, w2);
    std::cout << "Original vector  "<<dg::blas2::dot( v2, w2d, v2) << "\n";
    std::cout << "Projected vector "<<dg::blas2::dot( w2, w2dn, w2) << "\n";
    std::cout << "Difference       "<<dg::blas2::dot( v2, w2d, v2) - dg::blas2::dot( w2, w2dn, w2) << "\n" << std::endl;

    std::cout << "TEST GCD AND LCM\n";
    std::cout << "gcd of 1071 and 462 is "<<dg::gcd( 1071, 462)<<" (21)\n";
    std::cout << "lcm of 1071 and 462 is "<<dg::lcm( 1071, 462)<<" (23562)\n"<<std::endl;
    std::cout << "TEST OF DIFFERENCE\n";
    dg::DifferenceNorm<dg::HVec> diff( g2, g2n);
    std::cout << diff( v2, w2)<<" (should be zero) \n";
    dg::HVec x2 = evaluate( sine, g2);
    dg::HVec y2 = evaluate( sine, g2n);
    std::cout << diff( x2, y2)<<" \n";

    return 0;
}
