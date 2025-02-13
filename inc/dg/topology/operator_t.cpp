#include <iostream>
#include "operator.h"
#include "catch2/catch_all.hpp"

TEST_CASE( "SquareMatrix")
{

    SECTION( "Misc")
    {
        dg::SquareMatrix<double> lilj = dg::create::lilj<double>(3);
        dg::SquareMatrix<double> pidxpj = dg::create::pidxpj<double>(3);
        dg::SquareMatrix<double> rilj = dg::create::rilj<double>(3);

        dg::SquareMatrix<double> op = lilj + pidxpj;
        //dg::SquareMatrix<double> op = pidxpj - pidxpj.transpose();
        //op(0,0) = 1, op(0,1) = 2, op(0,2) = 0;
        //op(1,0) = 2, op(1,1) = 4, op(1,2) = 1;
        //op(2,0) = 2, op(2,1) = 1, op(2,2) = 0;
        INFO( "SquareMatrix\n"<<op);
        std::vector<double> result = {1,1,1, -1,1,1, 1,-1,1};
        CHECK( result == op.data());
        std::vector<unsigned> pivot( op.size());
        dg::SquareMatrix<double> lu(op);
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
        dg::SquareMatrix<double> inv_op = dg::create::inverse( op);
        INFO( "Inverse SquareMatrix\n"<<inv_op);
        result = {0.5,-0.5,0, 0.5,0,-0.5, 0,0.5,0.5};
        CHECK( inv_op.data() == result);
        auto delta = inv_op*op;
        INFO( "Multiplication\n"<<delta);
        result = {1,0,0, 0,1,0, 0,0,1};
        CHECK( delta.data() == result);

        //op.zero();
        op(0,2) = op(1,1) = op(2,0) = 0;// op(3,3)= 1;
        INFO( "SquareMatrix\n"<<op);
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
        INFO( "Inverse SquareMatrix\n"<<inv_op);
        result = {0.5,-0.5,0.5, 0.5,0.5,-0.5, 0.5,0.5,0.5};
        CHECK( inv_op.data() == result);

        delta = inv_op*op;
        INFO( "Multiplication\n"<<delta);
        result = {1,0,0, 0,1,0, 0,0,1};
        CHECK( delta.data() == result);
    }
    SECTION( "Matrix vector multiplication")
    {
        //! [matvec]
        auto mat = dg::SquareMatrix<double>({0,1,2, 3,4,5, 6,7,8});
        std::vector<double> vec = {1,2,3};
        std::vector<double> res = mat*vec;
        CHECK( res == std::vector<double>{8, 26, 44});
        //! [matvec]

        dg::blas1::copy( 0., res);
        dg::apply( mat, vec, res);
        CHECK( res == std::vector<double>{8, 26, 44});
    }
    SECTION( "symv 1")
    {
        //! [symv 1]
        auto mat = dg::SquareMatrix<double>({0,1,2, 3,4,5, 6,7,8});
        std::vector<double> vec = {1,2,3}, res(3);
        dg::blas2::symv( mat, vec, res);
        CHECK( res == std::vector<double>{8, 26, 44});
        //! [symv 1]
    }
    SECTION( "symv 2")
    {
        //! [symv 2]
        auto mat = dg::SquareMatrix<double>({0,1,2, 3,4,5, 6,7,8});
        std::vector<double> vec = {1,2,3}, res(3, 1000);
        dg::blas2::symv( 0.5, mat, vec, 2., res);
        CHECK( res == std::vector<double>{2004, 2013, 2022});
        //! [symv 2]
    }
}
