#include <iostream>
#include <cmath>

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


TEST_CASE( "dg::SparseTensor")
{
    const thrust::host_vector<double> zero(0,10), one(1,10), two(1,20),
       three(1,30), four(1,40), five(1,50), six(1,60), seven(1,70),
       eight(1,80), nine(1,90);
    dg::Grid2d grid( 0,1,0,1, 1, 10, 1); //grid with 10 elements
    dg::SparseTensor<thrust::host_vector<double> > dense2d(grid);
    dense2d.idx(0,0) = 0, dense2d.idx(0,1) = 1;
    dense2d.idx(1,0) = 1, dense2d.idx(1,1) = 2;
    dense2d.values().resize(3);
    dense2d.values()[0] = eight; dense2d.values()[1] = two; dense2d.values()[2] = nine;
    std::vector<double> result = {80,20,80, 20,90,80, 80,80,20};
    INFO( "Dense 2d Tensor");
    REQUIRE( line_element(dense2d) == result);

    std::vector<thrust::host_vector<double> > values(4);
    values[0] = seven; values[1] = three; values[2] = nine; values[3] = one;
    dg::SparseTensor<thrust::host_vector<double> > sparse3d(zero);
    sparse3d.values() = values;
    sparse3d.idx(0,0) = 0, sparse3d.idx(0,1) = 1                       ;
    sparse3d.idx(1,0) = 1                       , sparse3d.idx(1,2) = 3;
    sparse3d.idx(2,0) = 1                       , sparse3d.idx(2,2) = 3;
    dg::SparseTensor<thrust::device_vector<double> > sparse3d_D=sparse3d;

    INFO( "Sparse 3d Tensor");
    result = {70,30,70, 30,30,10, 30,70,10};
    CHECK( result == line_element( sparse3d));
    sparse3d = sparse3d_D;
    INFO( "Transfer device to host");
    CHECK( result == line_element( sparse3d));
    INFO( "transpose");
    result = {70,30,30, 30,30,70, 70,10,10};
    CHECK( result == line_element( sparse3d.transpose()));

    sparse3d.idx(2,0)=sparse3d.idx(0,2);
    sparse3d.idx(1,2)=sparse3d.idx(2,1);
    sparse3d.idx(1,1)=3;
    sparse3d.values()[3]=nine;
    result = {70,30,70, 30,90,70, 70,70,90};
    CHECK( result == line_element( sparse3d));

}
