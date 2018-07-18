#include <iostream>

#include "view.h"
#include "typedefs.h"
#include "../blas1.h"


int main()
{

    std::vector<double> test( 100, 3.);

    dg::View<dg::DVec> view( test.data(), test.size());
    dg::blas1::copy( 7., view);
    std::cout << "The original now has "<<test[0]<<" (7)\n";
    view.data() = &test[50];
    view.size() = 50;
    dg::blas1::copy( 3., view);
    std::cout << "The original now has "<<test[0]<<" (7) and "<<test[50]<<" (3)\n";
    const std::vector<double> const_test( 100, 42);
    const dg::View<const dg::DVec> const_view( const_test.data(), 50);
    dg::blas1::copy( const_view, view);
    std::cout << "The original now has "<<test[50]<<" (42)\n";

    return 0;
}
