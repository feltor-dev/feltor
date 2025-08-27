#include <iostream>
#include "functors.h"

#include "catch2/catch_all.hpp"

TEST_CASE( "phi functions")
{
    // Test correct evaluation of dividied differences formulas

    std::vector<double> limits = {1,0.5,1./6.,1./24.};
    std::vector<double> evals_small = {5e-11,1.66666666e-11,4.16666666e-12,8.3333333e-13};
    std::vector<double> evals_half = {5.0000166667e-6,1.6666708333e-6,4.1666750000e-7,8.33334722224206e-8};
    std::vector<double> evals_third = {0.0005001667083416681,0.0001667083416680558,0.00004167500138908733,8.334722420659725e-6};
    std::vector<double> evals_one = {0.7182818284590452,0.2182818284590452,0.05161516179237857,0.009948495125711902};
    std::vector<double> evals_ten = {2201.546579480672,219.6546579480672,21.79879912814005,2.138213246147338};

    std::vector<std::function<double(double)>> funcs = {dg::mat::phi1<double>, dg::mat::phi2<double>, dg::mat::phi3<double>, dg::mat::phi4<double>};
    unsigned u = GENERATE( 0, 1, 2, 3);
    INFO( "Phi"<<u+1);
    INFO( "Ana "<<limits[u] << "\nNum "<<funcs[u](0));
    CHECK( limits[u] == funcs[u](0));

    INFO( "Ana "<<evals_small[u] << "\nNum "<<funcs[u](1e-10)-funcs[u](0));
    INFO( "Relative Error " <<(evals_small[u] +funcs[u](0) -funcs[u](1e-10))/evals_small[u]);
    CHECK( (evals_small[u] +funcs[u](0) -funcs[u](1e-10))/evals_small[u] < 1e-15);

    INFO( "Ana "<<evals_half[u] << "\nNum "<<funcs[u](1e-5)-funcs[u](0));
    INFO( "Relative Error " <<(evals_half[u] +funcs[u](0) -funcs[u](1e-5))/evals_half[u]);
    CHECK( fabs(evals_half[u] +funcs[u](0) -funcs[u](1e-5))/evals_half[u] < 1e-15);

    INFO( "Ana "<<evals_third[u] << "\nNum "<<funcs[u](1e-3)-funcs[u](0));
    INFO( "Relative Error " <<(evals_third[u] +funcs[u](0) -funcs[u](1e-3))/evals_third[u]);
    CHECK( fabs(evals_third[u] +funcs[u](0) -funcs[u](1e-3))/evals_third[u] < 1e-11);

    INFO( "Ana "<<evals_one[u] << "\nNum "<<funcs[u](1.0)-funcs[u](0));
    INFO( "Relative Error " <<(evals_one[u] +funcs[u](0) -funcs[u](1.0))/evals_one[u]);
    CHECK( fabs(evals_one[u] +funcs[u](0) -funcs[u](1.0))/evals_one[u] < 1e-13);

    INFO( "Ana "<<evals_ten[u] << "\nNum "<<funcs[u](10)-funcs[u](0));
    INFO( "Relative Error " <<(evals_ten[u] +funcs[u](0) -funcs[u](10))/evals_ten[u]);
    CHECK( fabs(evals_ten[u] +funcs[u](0) -funcs[u](10))/evals_ten[u] < 1e-15 );
}
