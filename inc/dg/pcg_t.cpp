
#include <iostream>
#include "pcg.h"
#include "topology/operator.h"
#include "catch2/catch_all.hpp"


TEST_CASE( "PCG real")
{

    SECTION( "Set and get max")
    {
        unsigned n=10;
        dg::PCG pcg( std::vector( 10, 1), n);
        CHECK( pcg.get_max() == n);
        pcg.set_max( 3);
        CHECK( pcg.get_max() == 3);
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

        std::vector<double> b = rhs, x = rhs;
        dg::PCG pcg( rhs, n);
        unsigned num_steps = pcg.solve( t, x, b, 1., 1.);
        INFO( "Num steps "<<num_steps);
        CHECK( num_steps < n);
        for( unsigned u=0; u<n; u++)
        {
            INFO( "Solution ("<<u<<") "<<x[u]<<" "<<sol[u]<<" diff "<<x[u]-sol[u]);
            CHECK( fabs( x[u] - sol[u]) < 1e-12);
        }
    }

}
TEST_CASE( "PCG complex")
{
    SECTION( "Invert Hermitian matrix")
    {
        // Computed Solution with Mathematica
        double a = 1, b = 2, c = 3, d = 4, m = 5, n = 6;
        std::vector< thrust::complex<double>> data = {0, {a, -b},{c, -d}, {a, b}, 1, {m, -n}, {c, d}, {m, n}, 2};
        dg::SquareMatrix<thrust::complex<double>> H(data.begin(), data.end());
        std::vector <thrust::complex<double>> rhs = {{1,1}, {2,-3}, {4,5}}, x = {{0,1},{0,1},{0,1}};
        std::vector <thrust::complex<double>> sol = {{41./51., - 88./17.}, {23./51., 173./51.}, {-26./51., - 20./17.}};
        dg::PCG pcg( x, 4);
        unsigned num_steps = pcg.solve( H, x, rhs, 1., 1., 1e-8);
        INFO( "Num steps "<<num_steps);
        CHECK( num_steps <= 3);
        for( unsigned u=0; u<3; u++)
        {
            INFO( "Solution ("<<u<<") "<<x[u]<<" "<<sol[u]<<" diff "<<x[u]-sol[u]);
            CHECK( norm( x[u] - sol[u]) < 1e-12);
        }
    }
    SECTION( "Invert Complex Symmetric matrix")
    {
        // Computed Solution with Mathematica
        double a = 1, b = 2, c = 3, d = 4, m = 5, n = 6;
        std::vector< thrust::complex<double>> data = {0, {a, b},{c, d}, {a, b}, 1, {m, n}, {c, d}, {m, n}, 2};
        dg::SquareMatrix<thrust::complex<double>> H(data.begin(), data.end());
        std::vector <thrust::complex<double>> rhs = {{1,1}, {2,-3}, {4,5}}, x = {{0,1},{0,1},{0,1}};
        std::vector <thrust::complex<double>> sol = {{-7905./24713., -8588./24713.}, {26913./24713., 6251./24713.}, {-4422./24713., - 5892./24713.}};
        dg::COCG<std::vector<thrust::complex<double>>> pcg( x, 40);
        unsigned num_steps = pcg.solve( H, x, rhs, 1., 1., 1e-8);
        INFO( "Num steps "<<num_steps);
        CHECK( num_steps <= 3);
        for( unsigned u=0; u<3; u++)
        {
            INFO( "Solution ("<<u<<") "<<x[u]<<" "<<sol[u]<<" diff "<<x[u]-sol[u]);
            CHECK( norm( x[u] - sol[u]) < 1e-12);
        }
    }
}
