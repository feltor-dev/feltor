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
    //Projection might not be correct any more due to layout change
    std::cout << "TEST 1D\n";
    unsigned n_old = 4, n_new = 3, N = 10, Nf = 1;
    dg::Grid1d<double> go ( 0, M_PI, n_old, N);
    dg::Grid1d<double> gn ( 0, M_PI, n_new, N*Nf);
    cusp::coo_matrix<int, double, cusp::host_memory> proj = dg::create::projection1d( go, gn);
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

    /*
    std::cout << "TEST KRONECKER PRODUCT\n";
    dg::create::detail::HelperMatrix<double> m1(2,2), m2(2,2);
    for( unsigned i=0; i<2;i++)
        for( unsigned j=0; j<2;j++)
            m1(i,j) = 2*i+j+1;
    m2(0,0) = 0; m2( 0, 1) = 5; m2( 1,0) = 6, m2( 1,1) = 7;
    std::cout << "M1 \n"<<m1 << "Times\n"<<m2;
    std::cout << "Is \n"<<dg::create::detail::kronecker( m1, m2);
    std::cout << "(Compare Wikipedia for correctness!)\n"<<std::endl;
    */
    /*
    std::cout << "TEST GCD AND LCM\n";
    std::cout << "gcd of 1071 and 462 is "<<dg::gcd( 1071, 462)<<" (21)\n";
    std::cout << "lcm of 1071 and 462 is "<<dg::lcm( 1071, 462)<<" (23562)\n"<<std::endl;
    */

    std::cout << "TEST 2D\n";
    n_old = 7, n_new = 3, N = 20, Nf = 1;
    dg::Grid2d<double> g2o (0, M_PI, 0, M_PI, n_old, N, N);
    dg::Grid2d<double> g2n (0, M_PI, 0, M_PI, n_new, N, N*Nf);
    cusp::coo_matrix<int, double, cusp::host_memory> proj2d = dg::create::projection2d( g2o, g2n);
    const dg::HVec sinO = dg::evaluate( sine, g2o), 
                   sinN = dg::evaluate( sine, g2n);
    dg::HVec w2do = dg::create::weights( g2o);
    dg::HVec w2dn = dg::create::weights( g2n);
    dg::HVec sinP( g2n.size());
    dg::blas2::gemv( proj2d, sinO, sinP);
    std::cout << "Original vector  "<<dg::blas2::dot( sinO, w2do, sinO) << "\n";
    std::cout << "Projected vector "<<dg::blas2::dot( sinP, w2dn, sinP) << "\n";
    std::cout << "Evaluated vector "<<dg::blas2::dot(sinN, w2dn, sinN) << "\n";
    std::cout << "Difference       "<<dg::blas2::dot( sinO, w2do, sinO) - dg::blas2::dot( sinP, w2dn, sinP) << "\n" << std::endl;

    std::cout << "TEST OF DIFFERENCE\n";
    dg::DifferenceNorm<dg::HVec> diff( g2o, g2n);
    std::cout << "Information loss due to projection:\n";
    std::cout << diff( sinO, sinP)<<" (should converge to zero) \n";
    std::cout << "Difference between two grid evaluations:\n";
    std::cout << diff( sinO, sinN)<<" (should converge to zero!) \n";
    std::cout << "Difference between projection and evaluation      \n";
    dg::blas1::axpby( 1., sinN, -1., sinP);
    std::cout << dg::blas2::dot( sinP, w2dn, sinP)<<" (smaller than above)\n";


    return 0;
}
