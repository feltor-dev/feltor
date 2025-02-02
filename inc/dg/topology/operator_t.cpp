#include <iostream>
#include "operator.h"
#include "catch2/catch.hpp"

TEST_CASE( "Operator")
{

    dg::Operator<double> lilj = dg::create::lilj<double>(3);
    dg::Operator<double> pidxpj = dg::create::pidxpj<double>(3);
    dg::Operator<double> rilj = dg::create::rilj<double>(3);

    dg::Operator<double> op = lilj + pidxpj;
    //dg::Operator<double> op = pidxpj - pidxpj.transpose();
    //op(0,0) = 1, op(0,1) = 2, op(0,2) = 0;
    //op(1,0) = 2, op(1,1) = 4, op(1,2) = 1;
    //op(2,0) = 2, op(2,1) = 1, op(2,2) = 0;
    INFO( "Operator\n"<<op);
    std::vector<double> result = {1,1,1, -1,1,1, 1,-1,1};
    CHECK( result == op.data());
    std::vector<unsigned> pivot( op.size());
    dg::Operator<double> lu(op);
    double det= dg::create::lu_pivot( lu, pivot);
    INFO( "Determinant "<<det);
    CHECK( det == 4);
    INFO( "lu decomposition\n"<<lu);
    result = {1,1,1, -1,2,2, 1,-1,2};
    CHECK( lu.data() == result);
    INFO( "pivot ");
    for( unsigned i=0; i<op.size(); i++)
        INFO( pivot[i] <<" ");
    CHECK( pivot == std::vector<unsigned>{0,1,2});
    dg::Operator<double> inv_op = dg::create::inverse( op);
    INFO( "Inverse Operator\n"<<inv_op);
    result = {0.5,-0.5,0, 0.5,0,-0.5, 0,0.5,0.5};
    CHECK( inv_op.data() == result);
    auto delta = inv_op*op;
    INFO( "Multiplication\n"<<delta);
    result = {1,0,0, 0,1,0, 0,0,1};
    CHECK( delta.data() == result);


    //op.zero();
    op(0,2) = op(1,1) = op(2,0) = 0;// op(3,3)= 1;
    INFO( "Operator\n"<<op);
    result = {1,1,0, -1,0,1, 0,-1,1};
    CHECK( op.data() == result);
    inv_op = dg::create::inverse(op);
    lu = op;
    det= dg::create::lu_pivot( lu, pivot);
    INFO( "Determinant "<<det);
    CHECK( det == 2);
    INFO( "lu decomposition\n"<<lu);
    result = {1,1,0, -1,1,1, 0,-1,2};
    CHECK( lu.data() == result);
    INFO( "pivot ");
    for( unsigned i=0; i<op.size(); i++)
        INFO( pivot[i] <<" ");
    CHECK( pivot == std::vector<unsigned>{0,1,2});
    INFO( "Inverse Operator\n"<<inv_op);
    result = {0.5,-0.5,0.5, 0.5,0.5,-0.5, 0.5,0.5,0.5};
    CHECK( inv_op.data() == result);

    delta = inv_op*op;
    INFO( "Multiplication\n"<<delta);
    result = {1,0,0, 0,1,0, 0,0,1};
    CHECK( delta.data() == result);
}
