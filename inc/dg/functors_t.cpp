#include <iostream>


#include "functors.h"

#include "catch2/catch_all.hpp"

//TODO Add tests for other functors

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
