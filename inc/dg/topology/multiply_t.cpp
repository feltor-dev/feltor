#include <iostream>
#include <cmath>

#include "../backend/typedefs.h"
#include "tensor.h"
#include "multiply.h"

#include "catch2/catch_all.hpp"

template<class container>
std::vector<double> line_element( const dg::SparseTensor<container >& t)
{
    std::vector<double> v;
    for( unsigned i=0; i<3; i++)
    for( unsigned j=0; j<3; j++)
        v.push_back( t.value(i,j)[0]);
    return v;

}


TEST_CASE( "dg::tensor")
{
    const thrust::host_vector<double> one(1,1), two(1,2), three(1,3),
       four(1,4), five(1,5), six(1,6), seven(1,7), eight(1,8), nine(1,9),
       zero(1,0);
    dg::SparseTensor<thrust::host_vector<double> > t;
    t.idx(0,0) = 2, t.idx(0,1) = 3, t.idx(0,2) = 0;
    t.idx(1,0) = 3, t.idx(1,1) = 4, t.idx(1,2) = 0;
    t.idx(2,0) = 0, t.idx(2,1) = 0, t.idx(2,2) = 1;
    t.values().resize(5);
    t.values()[0] = zero, t.values()[1] = one, t.values()[2]= two,
        t.values()[3] = three, t.values()[4]=four;
    thrust::host_vector<double> inout0=eight, inout1=nine, inout2=two,
        work0(inout0), work1(inout1), work2(inout2);
    thrust::host_vector<double> mu(five), nu;
    SECTION( "Basic ops")
    {
        std::vector<double> result = {2,3,0, 3,4,0, 0,0,1};
        INFO("Begin T");
        REQUIRE( line_element(t) == result);
        INFO( "Scale with 5");
        dg::tensor::scal(t,mu);
        result = {10,15,0, 15,20,0, 0,0,5};
        CHECK( line_element(t) == result);
        INFO( "Scale with empty element");
        dg::tensor::scal(t,1.);
        CHECK( line_element(t) == result);
        INFO( "Scale with 1/5");
        dg::tensor::scal(t, 1./5.);
        result = {2,3,0, 3,4,0, 0,0,1};
        CHECK( line_element(t) == result);
        INFO( "Scale with container(2)");
        dg::tensor::scal(t,two);
        result = {4,6,0, 6,8,0, 0,0,2};
        CHECK( line_element(t) == result);
        INFO( "explicit dense Tensor");
        dg::SparseTensor<thrust::host_vector<double> > dense3d = t;
        CHECK( line_element(dense3d) == result);
    }

    SECTION( "Test Tensor multiplies")
    {
        std::vector<double> result = {4,6,0, 6,8,0, 0,0,2};
        dg::tensor::scal(t,two);
        REQUIRE( line_element(t) == result);
        INFO( "Multiply T with [8,9]");
        dg::tensor::multiply2d( t, eight, nine, work0, work1);
        INFO( "Result         is ["<<work0[0]<<" "<<work1[0]<<"] ([86 120])");
        CHECK( work0[0] == 86);
        CHECK( work1[0] == 120);
        INFO( "Scalar product 2d");
        inout0 = eight;
        dg::tensor::scalar_product2d( 1., 2., one, two, t, 2., eight, nine, 1.,
                inout0);
        INFO( "Result         is "<<inout0[0]<<" (1312)");
        CHECK( inout0[0] == 1312);
        INFO( "Multiply T^{-1} with [86,120]");
        dg::tensor::inv_multiply2d(1., t, work0, work1, 0., work0, work1);
        INFO( "Result         is ["<<work0[0]<<" "<<work1[0]<<"] ([8 9])");
        CHECK( work0[0] == 8);
        CHECK( work1[0] == 9);
        inout0=eight, inout1=nine, inout2=two;
        dg::tensor::multiply2d(1., t, inout0, inout1, 0., work0, inout1);
        INFO( "Result inplace is ["<<work0[0]<<" "<<inout1[0]<<"] ([86 120])");
        CHECK( work0[0] == 86);
        CHECK( inout1[0] == 120);
        INFO( "Modify and swap idx");
        t.idx(0,2) = 4;
        std::swap( t.idx(1,1), t.idx(2,1));
        result = {4,6,0, 6,0,0, 0,8,2};
        INFO( "Multiply T with [8,9,2]");
        dg::tensor::multiply3d(t, eight, nine,two, work0, work1, work2);
        INFO( "Result         is ["<<work0[0]<<" "<<work1[0]<<" "<<work2[0]
            <<"] ([102 48 76])");
        CHECK( work0[0] == 102);
        CHECK( work1[0] ==  48);
        CHECK( work2[0] ==  76);
        INFO( "Scalar product 3d");
        inout0 = eight;
        dg::tensor::scalar_product3d( 1., 3., one, two,three, t, 3., 8.,9.,2.,
            -100., inout0);
        INFO( "Result         is "<<inout0[0]<<" (3034)");
        CHECK( inout0[0] == 3034);
        INFO( "Multiply T^{-1} with [102,48,76]");
        dg::tensor::inv_multiply3d(1., t, work0, work1, work2, 0., work0,
            work1, work2);
        INFO( "Result         is ["<<work0[0]<<" "<<work1[0]<<" "<<work2[0]
            <<"] ([8 9 2])");
        CHECK( work0[0] == 8);
        CHECK( work1[0] == 9);
        CHECK( work2[0] == 2);
        inout0=eight, inout1=nine, inout2=two;
        dg::tensor::multiply3d(1., t, inout0, inout1, inout2, 0., inout0,
            inout1, inout2);
        INFO( "Result inplace is ["<<inout0[0]<<" "<<inout1[0]<<" "
            <<inout2[0]<<"] ([102 48 76])");
        CHECK( inout0[0] == 102);
        CHECK( inout1[0] ==  48);
        CHECK( inout2[0] ==  76);

        double det = dg::tensor::determinant(t)[0];
        INFO( "Determinant3d of T: "<<det<<" (312)");
        CHECK( det == 312);
        double det2d = dg::tensor::determinant2d(t)[0];
        INFO( "Determinant2d of T: "<<det2d<<" (-36)");
        CHECK( det2d == -36);
    }
}

TEST_CASE( "Documentation")
{
    SECTION( "SparseTensor")
    {
        //! [sparse tensor]
        dg::SparseTensor<dg::HVec> metric; // allocate 3x3 index matrix
        metric.idx(0,0) = 1, metric.idx(0,1) = 0, metric.idx(0,2) = 0;
        metric.idx(1,0) = 0, metric.idx(1,1) = 2, metric.idx(1,2) = 0;
        metric.idx(2,0) = 0, metric.idx(2,1) = 0, metric.idx(2,2) = 3;
        std::vector<dg::HVec> values( 4);
        values[0] = dg::HVec( 100, 0);   // the zero element
        values[1] = dg::HVec( 100, 20.); // construct gxx element
        values[2] = dg::HVec( 100, 30.); // construct gyy element
        values[3] = dg::HVec( 100, 1.); // construct gzz element
        metric.values() = values;
        // then we can for example use dg::tensor functions:
        dg::HVec det = dg::tensor::determinant( metric);
        CHECK( det == dg::HVec( 100, 20*30));
        // the individual elements can be accessed via the access operator
        dg::HVec gxx = metric.value(0,0);
        CHECK( gxx == dg::HVec( 100, 20));
        //! [sparse tensor]
    }
}
