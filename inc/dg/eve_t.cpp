
#include <iostream>
#include "eve.h"
#include "topology/operator.h"
#include "catch2/catch_all.hpp"


TEST_CASE( "EVE real")
{

    SECTION( "Set and get max")
    {
        unsigned n=10;
        dg::EVE eve( std::vector( 10, 1), n);
        CHECK( eve.get_max() == n);
        eve.set_max( 3);
        CHECK( eve.get_max() == 3);
    }
    SECTION( "Invert almost singular")
    {
        // This was tested for SquareMatrix in operator_t
        unsigned n = 10;
        dg::SquareMatrix<double> t( n, 1.);
        double eps = 1e-3;
        for( unsigned u=0; u<n; u++)
            t(u,u) = 1.0 + eps;

        std::vector<unsigned> p;
        auto lu = t;
        // Det is ~ 1e-26
        dg::create::lu_pivot( lu, p);
        std::vector<double> rhs(n, 1.), sol( rhs);
        dg::lu_solve( lu, p, sol);

        std::vector<double> b = rhs, x = std::vector<double>( n, 0.);
        dg::EVE eve( rhs, n);

        double eve_max;
        unsigned num_steps = eve.solve( t, x, b, 1., 1., eve_max, 1e-10);
        INFO( "Num steps "<<num_steps<<" EV max "<<eve_max);
        CHECK( num_steps < n);
        // From Mathematica max EV is known
        CHECK( fabs(eve_max  - (double)(n+eps)) < 1e-14);
        for( unsigned u=0; u<n; u++)
        {
            INFO( "Solution ("<<u<<") "<<x[u]<<" "<<sol[u]<<" diff "<<x[u]-sol[u]);
            CHECK( fabs( x[u] - sol[u]) < 1e-12);
        }
    }

}
