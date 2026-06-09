#include <iostream>


#include "functors.h"

#include "catch2/catch_all.hpp"

TEST_CASE("Basic Functors")
{
    std::vector<double> xs( { 1.,2.,3.,4.});
    SECTION("Random")
    {
        dg::RandomNumbers<float> rand( 0,1);
        for( unsigned u=0; u<xs.size(); u++)
        {
            // Manually check that random numbers look random
            //std::cout << rand(xs[u], xs[u], xs[u], xs[u])<<"\n";

            CHECK( 0<=rand( xs[u], xs[u], xs[u]));
            CHECK( rand( ) <1);
        }
        // Test that rand can be called on device
        thrust::device_vector<float> xd( {1,2,3,4});
        thrust::transform( xd.begin(), xd.end(), xd.begin(), rand);
        for( unsigned u=0; u<xd.size(); u++)
        {
            CHECK( 0<=xd[u]);
            CHECK( xd[u] <1);
        }
    }

}

TEST_CASE("Fourier Series")
{
    // Taken from the axis of a W7-X vmec file

    const unsigned nfp = 5;
    const double phi = 0.5;

    SECTION("COSINE")
    {
        const std::vector<double> raxis_cc = {5.64539273e+00,  3.51552974e-01,
        1.22083861e-02,  6.20226702e-04, -3.23533404e-04, -1.51678453e-05,
        -8.81100973e-05,  6.59075599e-05, 2.08857013e-05, -6.94325109e-06,
        -1.48425226e-06,  1.17023571e-06, 8.29344548e-07};
        const double raxis = 5.367776503087852;
        dg::RealFourier1d raxis_f( raxis_cc, raxis_cc.size(), 0, 2.*M_PI/(double)nfp);
        double raxis_num = raxis_f( phi);
        INFO( "raxis num " << raxis_num);
        CHECK( fabs( raxis - raxis_num) < 1e-8);
    }
    SECTION("SINE")
    {
        std::vector<double> zaxis_cs = {-2.95355961e-01, -1.34627261e-02,
        -1.21378487e-04, 5.46036476e-05, 5.34576546e-05,  3.07960230e-05,
        7.57340608e-05, 6.57699520e-06, 5.29530210e-06,  1.46030535e-06,
        2.22207868e-07, 7.06320086e-08};
        dg::blas1::scal( zaxis_cs, -1);
        const double zaxis = 0.16405022896331034;
        dg::RealFourier1d zaxis_f( zaxis_cs, 0, zaxis_cs.size(), 2.*M_PI/(double)nfp);
        double zaxis_num = zaxis_f( phi);
        INFO( "zaxis num " << zaxis_num);
        CHECK( fabs( zaxis - zaxis_num) < 1e-8);
    }
    SECTION("const")
    {
        std::vector<double> ones( 1 + 12 + 12, 1);
        dg::RealFourier1d delta_f( ones, 13, 12, 2.*M_PI/(double)nfp);
        const double delta = 0.059471471292880385;
        double delta_num = delta_f(phi);
        INFO( "delta num "<<delta_num);
        CHECK( fabs( delta - delta_num ) < 1e-8);
    }
}

TEST_CASE( "Horner")
{
    SECTION("Horner1d")
    {
        // Test Legendre polynomial
        std::vector<double> c = {-63, 0., 3465, 0, -30030, 0, 90090, 0, -109395, 0, 46189};
        dg::blas1::scal( c, 1.0/256.);
        dg::Horner1d legendre( c);
        const double leg10 = -0.122124997387109375;
        double leg10_num = legendre(0.1);
        CHECK( fabs ( leg10_num - leg10) < 1e-12);
    }

}
