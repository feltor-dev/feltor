#include <iostream>


#include "dlt.h"
#include "functions.h"
#include "operator_tensor.h"
#include "catch2/catch_all.hpp"

TEST_CASE( "SquareMatrix tensorproduct")
{
    dg::SquareMatrix<double> Op1 = dg::create::delta<double>( 3);
    dg::SquareMatrix<double> Op2 = dg::DLT<double>::forward(3);
    auto op = dg::tensorproduct( Op1, Op2);
    for( int i=0; i<3; i++)
    for( int k=0; k<3; k++)
    {
        CHECK( op(i+0,k+0) == Op2(i,k));
        CHECK( op(i+0,k+3) == 0);
        CHECK( op(i+0,k+6) == 0);

        CHECK( op(i+3,k+0) == 0);
        CHECK( op(i+3,k+3) == Op2(i,k));
        CHECK( op(i+3,k+6) == 0);

        CHECK( op(i+6,k+0) == 0);
        CHECK( op(i+6,k+3) == 0);
        CHECK( op(i+6,k+6) == Op2(i,k));
    }

    auto test2 = dg::tensorproduct( 2, Op2);
    std::vector<int> row = {0,3,6,9,12,15,18};
    std::vector<int> col = {0,1,2,0,1,2,0,1,2,3,4,5,3,4,5,3,4,5};
    thrust::host_vector<double> val( 18);
    thrust::copy( Op2.data().begin(), Op2.data().end(), val.begin());
    thrust::copy( Op2.data().begin(), Op2.data().end(), val.begin()+9);
    CHECK( row == test2.row_offsets());
    CHECK( col == test2.column_indices());
    CHECK( val == test2.values());
}
