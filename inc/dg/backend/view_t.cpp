#include <iostream>

#include "view.h"
#include "typedefs.h"
#include "../blas1.h"

#include "catch2/catch_all.hpp"


TEST_CASE( "The view class")
{

    std::vector<double> test( 100);
    for( unsigned i=0; i<100; i++)
        test[i] = i;

    dg::View<dg::HVec> view( test.data(), test.size());
    INFO("The view has size "<<view.size()<<" (100)\n");
    CHECK( view.size() == 100);
    dg::blas1::plus( view, 7);
    INFO("The original now has "<<test[0]<<" (7)\n");
    CHECK( test[0] == 7);
    view.construct( &test[50], 50);
    dg::blas1::plus( view, 3);
    // test now contains original plus 10
    INFO("The original now has "<<test[0]<<" (7) and "<<test[50]<<" (60)\n");
    CHECK( test[0] == 7);
    CHECK( test[50] == 60);
    std::vector<double> const_test( 75);
    for( unsigned i=0; i<75; i++)
        const_test[i] = i;
    const dg::View<const dg::HVec> const_view( const_test.data(), 50);
    dg::HVec construct = dg::construct<dg::HVec>( const_view);
    INFO("Constructed vector has "<<construct.size()<<" (50) elements\n");
    CHECK( construct.size() == 50);
    INFO("Constructed vector element "<<construct[41]<<" (41)\n");
    CHECK( construct[41] == 41);
    dg::HVec assign;
    dg::assign( const_view, assign);
    INFO("Assigned vector element "<<assign[42]<<" (42)\n");
    CHECK( assign[42] == 42);
    INFO("Assigned vector has "<<construct.size()<<" (50) elements\n");
    CHECK( construct.size() == 50);

    INFO("Const View has "<<const_view.size()<<" (50) elements\n");
    CHECK( const_view.size() == 50);
    view = dg::View<dg::HVec>( test.data(), const_view.size());
    dg::blas1::copy( const_view, view);
    INFO("The original now has "<<test[49]<<" (49)\n");
    CHECK( test[49] == 49);
}
