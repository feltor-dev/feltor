#include <iostream>
#include <cusp/print.h>
#include "dg/blas.h"
#include "derivatives.h"
#include "projection.h"
#include "evaluation.h"
#include "fast_interpolation.h"

double sine( double x){ return sin(x);}
double sine( double x, double y){return sin(x)*sin(y);}
double sine( double x, double y, double z){return sin(x)*sin(y);}
//Actually this file is a test for fast_interpolation

int main()
{
    std::array<unsigned,2> ns = {3,9}, Ns = {20,40};
    for( unsigned i=0; i<2; i++)
    {
        std::cout << "TEST 1D\n";
        unsigned n_old = ns[i], n_new = 3, N_old = Ns[i], N_new = 20;
        //std::cout << "Type n and N of old (fine) grid!\n";
        //std::cin >> n_old >> N_old;
        //std::cout << "Type n and N of new (coarser) grid!\n";
        //std::cin >> n_new >> N_new;
        std::cout << "Fine   Grid "<< n_old << " x "<<N_old <<"\n";
        std::cout << "Coarse Grid "<< n_new << " x "<<N_new <<"\n";
        dg::Grid1d go ( 0, M_PI/2., n_old, N_old);
        dg::Grid1d gn ( 0, M_PI/2., n_new, N_new);
        dg::DMatrix proj = dg::create::fast_projection1d( go, n_old/n_new,  N_old/N_new);
        dg::DMatrix inte = dg::create::fast_interpolation1d( gn, n_old/n_new, N_old/N_new);
        dg::DVec v = dg::evaluate( sine, go);
        dg::DVec w1do = dg::create::weights( go);
        dg::DVec w1dn = dg::create::weights( gn);
        dg::DVec w( gn.size());
        dg::blas2::symv( proj, v, w);
        std::cout << "Original vector  "<<dg::blas1::dot( w1do, v) << "\n";
        std::cout << "Projected vector "<<dg::blas1::dot( w1dn, w) << "\n";
        std::cout << "Difference       "<<dg::blas1::dot( w1do, v) - dg::blas1::dot( w1dn, w) << " (Must be 0)\n"<<std::endl;
        w = dg::evaluate( sine, gn);
        dg::blas2::symv( inte, w, v);
        std::cout << "Original vector  "<<dg::blas1::dot( w1dn, w) << "\n";
        std::cout << "Interpolated vec "<<dg::blas1::dot( w1do, v) << "\n";
        std::cout << "Difference       "<<dg::blas1::dot( w1do, v) - dg::blas1::dot( w1dn, w) << " (Must be 0)\n"<<std::endl;
        dg::DVec wP( w);
        dg::blas2::symv( proj, v, wP);
        dg::blas1::axpby( 1., wP, -1., w);
        std::cout << "Difference PI    "<<dg::blas2::dot( w, w1dn, w) << " (Must be 0)\n"<<std::endl;
        dg::HVec xvec = dg::evaluate( dg::cooX1d, go);
        dg::IDMatrix inte_m = dg::create::interpolation( xvec, gn, dg::DIR, "linear");
        std::cout <<" LINEAR interpolation: \n";
        w = dg::evaluate( sine, gn);
        dg::blas2::symv( inte_m, w, v);
        std::cout << "Original vector  "<<dg::blas1::dot( w1dn, w) << "\n";
        std::cout << "Interpolated vec "<<dg::blas1::dot( w1do, v) << "\n";
        std::cout << "Difference       "<<dg::blas1::dot( w1do, v) - dg::blas1::dot( w1dn, w) << " (Must be 0)\n"<<std::endl;
        dg::blas2::symv( proj, v, wP);
        dg::blas1::axpby( 1., wP, -1., w);
        std::cout << "Difference PI    "<<dg::blas2::dot( w, w1dn, w) << " (Must be 0)\n"<<std::endl;

        std::cout << "TEST 2D and 3D\n";

        dg::Grid3d g2o (0, M_PI, 0, M_PI, 0,1, n_old, N_old, N_old, 4);
        dg::Grid3d g2n (0, M_PI, 0, M_PI, 0,1, n_new, N_new, N_new, 4);
        cusp::coo_matrix<int, double, cusp::host_memory> inte2d = dg::create::interpolation( g2n, g2o);
        auto proj2d = dg::create::fast_projection( g2o, n_old/n_new, N_old/N_new, N_old/N_new);
        auto fast_inte2d = dg::create::fast_interpolation( g2n, n_old/n_new, N_old/N_new, N_old/N_new);
        auto forward = dg::create::fast_transform( dg::DLT<double>::forward( n_old),
            dg::DLT<double>::forward( n_old), g2o);
        auto backward = dg::create::fast_transform( dg::DLT<double>::backward( n_old),
            dg::DLT<double>::backward( n_old), g2o);
        const dg::HVec sinO( dg::evaluate( sine, g2o)),
                                    sinN( dg::evaluate( sine, g2n));
        dg::HVec w2do = dg::create::weights( g2o);
        dg::HVec w2dn = dg::create::weights( g2n);
        dg::HVec sinP( sinN), sinI(sinO), sinF(sinO);
        dg::blas2::gemv( proj2d, sinO, sinP); //FAST PROJECTION
        double value0 = sqrt(dg::blas2::dot( sinO, w2do, sinO));
        std::cout << "Original vector     "<<value0 << "\n";
        double value1 = sqrt(dg::blas2::dot( sinP, w2dn, sinP));
        std::cout << "Projected vector    "<<value1 << "\n";
        std::cout << "Difference in Norms "<<value0-value1 << std::endl;
        dg::blas1::axpby( 1., sinN, -1., sinP);
        double value2 = sqrt(dg::blas2::dot( sinP, w2dn, sinP)/dg::blas2::dot(sinN, w2dn, sinN));
        std::cout << "Difference between projection and evaluation      "<<value2<<"\n";
        dg::blas2::gemv( inte2d, sinO, sinP);
        value1 = sqrt(dg::blas2::dot( sinP, w2dn, sinP));
        std::cout << "Interpolated vec    "<<value1 << "\n";
        value0 = sqrt(dg::blas2::dot( sinO, w2do, sinO));
        std::cout << "Difference in Norms "<<value0 - value1 << "\n" << std::endl;
        dg::blas2::gemv( fast_inte2d, sinN, sinI);
        value2 = sqrt(dg::blas2::dot( sinI, w2do, sinI));
        std::cout << "Fast Interpolated vec "<< value2 << "\n";
        double value3 = sqrt(dg::blas2::dot( sinN, w2dn, sinN));
        std::cout << "Difference in Norms   "<<value2 - value3  << "\n" << std::endl;
        dg::blas2::gemv( forward, sinO, sinF);
        dg::blas2::gemv( backward, sinF, sinI);
        dg::blas1::axpby( 1., sinO, -1., sinI);
        value2 = sqrt(dg::blas2::dot( sinI, w2do, sinI));
        std::cout << "Forward-Backward Error   "<<value2 << " (Must be zero)\n" << std::endl;
    }
    std::cout << "Test backproject 1d\n";
    unsigned n=3, N = 20;
    dg::Grid1d g1d( 0.0, M_PI+0.0, n, N, dg::DIR);
    dg::Grid1d g1dequi( 0.0, M_PI, 1, n*N, dg::DIR);
    auto w1d = dg::create::weights( g1d);
    auto w1dequi = dg::create::weights( g1dequi);
    auto proj = dg::create::backproject( g1d);
    auto inv_proj = dg::create::inv_backproject( g1d);
    auto v = dg::evaluate( sine, g1d), w(v), x(v);
    dg::blas2::symv( proj, v, w);
    double integral = dg::blas1::dot( v, w1d);
    double integralequi = dg::blas1::dot( w, w1dequi);
    std::cout << "Error Integral is "<<(integral-integralequi)<<" (Must be zero)\n";
    dg::blas2::symv( inv_proj, w, x);
    dg::blas1::axpby( 1., v, -1., x);
    double err = dg::blas1::dot( x, x);
    std::cout << "Error is "<<sqrt(err)<<" (Must be zero)\n";

    std::cout << "Test backproject 2d\n";
    dg::Grid2d g2d(0., M_PI, 0., M_PI, 3, 10, 20);
    dg::Grid2d g2dequi = g2d;
    g2dequi.set( 1, g2d.shape(0), g2d.shape(1));
    auto w2d = dg::create::weights( g2d);
    auto w2dequi = dg::create::weights( g2dequi);
    proj = dg::create::backproject( g2d);
    inv_proj = dg::create::inv_backproject( g2d);
    v = dg::evaluate( sine, g2d), w=v, x=v;
    dg::blas2::symv( proj, v, w);
    integral = dg::blas1::dot( v, w2d);
    integralequi = dg::blas1::dot( w, w2dequi);
    std::cout << "Error Integral 2d is "<<(integral-integralequi)<<" (Must be zero)\n";
    dg::blas2::symv( inv_proj, w, x);
    dg::blas1::axpby( 1., v, -1., x);
    err = dg::blas1::dot( x, x);
    std::cout << "Error 2d is "<<sqrt(err)<<" (Must be zero)\n";
    return 0;
}
