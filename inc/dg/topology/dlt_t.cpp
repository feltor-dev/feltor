#include <iostream>
#include <iomanip>

#include "dlt.h"
#include "operator.h"

#include "catch2/catch_all.hpp"

TEST_CASE("DLT")
{
    using namespace Catch::Matchers;
    //Test if forward * backwards gives a Delta function and print abscissas and weights
    auto n = GENERATE( range<unsigned>( 1,21));
    //auto n = GENERATE( RangeGenerator<unsigned>( 1,21));

    INFO("# of polynomial coefficients n! \t"<<n);
    SECTION("Forward * Backward should give Delta");
    {
        dg::Operator<double> forward( dg::DLT<double>::forward(n));
        dg::Operator<double> backward( dg::DLT<double>::backward(n));
        auto delta = forward*backward;
        for( unsigned k=0; k<n; k++)
        for( unsigned l=0; l<n; l++)
        {
            if( k == l)
            {
                INFO( "Diagonal "<<k);
                CHECK( fabs(delta(k,l) -1)  < 1e-15);
            }
            else
            {
                INFO( "Off Diagonal "<<k<<" "<<l);
                CHECK( fabs(delta(k,l) )  < 1e-15);
            }
        }
    }
    SECTION( "Abscissas");
    {
        auto abs = dg::DLT<double>::abscissas(n);
        double sum =0;
        for( unsigned i=0; i<n; i++)
            sum += abs[i];
        INFO("Sum of all abscissas\t"<<sum<<" (0)");
        CHECK_THAT( sum, WithinAbs(  0, 1e-15));
    }
    SECTION( "Weights")
    {
        auto weights = dg::DLT<double>::weights(n);
        double sum = 0;
        for( unsigned i=0; i<n; i++)
            sum += weights[i];
        INFO("Sum of all weights\t"<<sum<<" (2)");
        CHECK_THAT( sum, WithinAbs(  2, 1e-15));
    }
}
