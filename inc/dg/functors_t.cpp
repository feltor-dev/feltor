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
