
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "tensor_traits_std.h"
#include "tensor_traits_thrust.h"
#include "sparsematrix.h"
#include "catch2/catch_all.hpp"

TEST_CASE("Format conversion")
{
    std::vector<int> coo_row = { 0 , 0, 1,1,1, 3, 4,4};
    std::vector<int> csr_row = dg::detail::coo2csr( 5, coo_row);
    CHECK( csr_row == std::vector{ 0, 2, 5, 5, 6, 8});
    auto coo = dg::detail::csr2coo( csr_row );
    CHECK( coo == coo_row);


}
TEST_CASE( "Construct sparse matrix")
{
    size_t num_rows = 3, num_cols = 5, num_nnz = 6;
    thrust::host_vector<int> rows(num_rows+1), cols(num_nnz);
    thrust::host_vector<double> vals(num_nnz);
    SECTION( "Construct")
    {
        dg::SparseMatrix<int,double,thrust::host_vector> mat ( num_rows, num_cols, rows, cols, vals);
        static_assert( std::is_same_v< dg::get_tensor_category< dg::SparseMatrix<
            int,double,thrust::host_vector>>, dg::SparseMatrixTag>);
        CHECK( mat.num_rows() == num_rows);
        CHECK( mat.num_cols() == num_cols);
        CHECK( mat.num_vals() == vals.size());
    }
    SECTION( "Set")
    {
        dg::SparseMatrix mat;
        mat.set( num_rows, num_cols, rows, cols, vals);
        CHECK( mat.num_rows() == num_rows);
        CHECK( mat.num_cols() == num_cols);
        CHECK( mat.num_vals() == vals.size());
    }
    SECTION( "Constructing sorts columns")
    {
        // 1 0 0 2 3
        // 2 0 5 0 0
        // 0 4 0 0 1
        unsigned num_rows = 3, num_cols = 5;
        std::vector<int> rows = {0,3,5,7}, cols = {4,3,0,2,0,4,1};
        std::vector<double> vals = {3,2,1,5,2,1,4};
        dg::SparseMatrix<int,double,std::vector> A ( num_rows, num_cols, rows, cols, vals);
        CHECK( A.row_offsets() == std::vector{0,3,5,7});
        CHECK( A.column_indices() == std::vector{0,3,4,0,2,1,4});
        CHECK( A.values() == std::vector<double>{1,2,3,2,5,4,1});
    }
    SECTION( "setFromCoo")
    {
        // 1 0 0 2 3
        // 2 0 5 0 0
        // 0 4 0 0 1
        unsigned num_rows = 3, num_cols = 5;
        std::vector<int> rows = {2,2,0,0,0,1,1}, cols = {4,1,4,3,0,2,0};
        std::vector<double> vals = {1,4, 3,2,1,5,2};
        dg::SparseMatrix<int,double,std::vector> A;
        A.setFromCoo( num_rows, num_cols, rows, cols, vals);
        CHECK( A.row_offsets() == std::vector{0,3,5,7});
        CHECK( A.column_indices() == std::vector{0,3,4,0,2,1,4});
        CHECK( A.values() == std::vector<double>{1,2,3,2,5,4,1});
    }

}
TEST_CASE( "Linear algebra")
{
    // 1 0 0 2 3
    // 2 0 5 0 0
    // 0 4 0 0 1
    unsigned num_rows = 3, num_cols = 5;
    std::vector<int> rows = {0,3,5,7}, cols = {0,3,4,0,2,1,4};
    std::vector<double> vals = {1,2,3,2,5,4,1};
    dg::SparseMatrix<int,double,std::vector> A ( num_rows, num_cols, rows, cols, vals);

    // 2 0 0 0
    // 0 3 3 5
    // 0 0 4 0
    // 0 0 0 5
    // 1 0 0 2
    num_rows = 5, num_cols = 4;
    rows = {0,1,4,5,6,8};
    cols = {0,1,2,3,2,3,0,3};
    vals = {2,3,3,5,4,5,1,2};
    dg::SparseMatrix<int,double,std::vector> B ( num_rows, num_cols, rows, cols, vals);
    static_assert( std::is_same_v <typename dg::SparseMatrix<int,double,std::vector>::policy, dg::SerialTag>);

    SECTION( "gemv")
    {
        std::vector<double> v(5,2), w(3,0);
        for( unsigned i=0; i<v.size(); i++)
            v[i] = double(i+1);

        w = A*v;
        CHECK( w[0] == 24);
        CHECK( w[1] == 17);
        CHECK( w[2] == 13);
    }
    SECTION( "gemm")
    {
        // 5 0 0 16
        // 4 0 20 0
        // 1 12 12 22
        auto C = A*B;
        CHECK( C.num_rows() == 3);
        CHECK( C.num_cols() == 4);
        CHECK( C.row_offsets() == std::vector<int>{ 0,2,4,8});
        CHECK( C.column_indices() == std::vector<int>{ 0,3,0,2, 0,1,2,3});
        CHECK( C.values() == std::vector<double>{ 5,16,4,20,1,12,12,22});
    }
    SECTION( "A * 1")
    {
        unsigned size = 5;
        std::vector<int> urows = {0,1,2,3,4,5}, ucols = {0,1,2,3,4};
        std::vector<double> uvals = {1., 1., 1., 1., 1.};
        dg::SparseMatrix<int,double,std::vector> unit5( size, size, urows, ucols, uvals);
        auto C = A*unit5;
        CHECK( C.num_rows() == 3);
        CHECK( C.num_cols() == 5);
        CHECK( C.row_offsets() == std::vector<int>{ 0,3,5,7});
        CHECK( C.column_indices() == std::vector<int>{ 0,3,4,0, 2,1,4});
        CHECK( C.values() == std::vector<double>{ 1,2,3,2,5,4,1});

    }
    SECTION( "1 * A")
    {
        unsigned size = 3;
        std::vector<int> urows = {0,1,2,3}, ucols = {0,1,2};
        std::vector<double> uvals = {1., 1., 1.};
        dg::SparseMatrix<int,double,std::vector> unit3( size, size, urows, ucols, uvals);
        auto C = unit3*A;
        CHECK( C.num_rows() == 3);
        CHECK( C.num_cols() == 5);
        CHECK( C.row_offsets() == std::vector<int>{ 0,3,5,7});
        CHECK( C.column_indices() == std::vector<int>{ 0,3,4,0, 2,1,4});
        CHECK( C.values() == std::vector<double>{ 1,2,3,2,5,4,1});

    }
    SECTION( "Addition")
    {
        // 0 2 0 0 4
        // 0 1 2 0 0
        // 0 0 0 0 0
        num_rows = 3, num_cols = 5;
        rows = {0,2,4,4}, cols = {1,4,1,2};
        vals = {2,4,1,2};
        dg::SparseMatrix<int,double,std::vector> D ( num_rows, num_cols, rows, cols, vals);

        // 1 2 0 2 7
        // 2 1 7 0 0
        // 0 4 0 0 1
        auto C = A + D;
        CHECK( C.num_rows() == 3);
        CHECK( C.num_cols() == 5);
        CHECK( C.row_offsets() == std::vector<int>{ 0,4,7,9});
        CHECK( C.column_indices() == std::vector<int>{ 0,1,3,4,0,1,2,1,4});
        CHECK( C.values() == std::vector<double>{ 1,2,2,7,2,1,7,4,1});
    }
    SECTION( "Scal")
    {

        // 0.5 0 0 1 1.5
        // 1 0 2.5 0 0
        // 0 2 0 0 0.5
        auto C = 0.5*A;
        CHECK( C.num_rows() == 3);
        CHECK( C.num_cols() == 5);
        CHECK( C.row_offsets() == std::vector<int>{ 0,3,5,7});
        CHECK( C.column_indices() == std::vector<int>{ 0,3,4,0,2,1,4});
        CHECK( C.values() == std::vector<double>{ 0.5,1,1.5,1,2.5,2,0.5});
    }
}
TEST_CASE( "SpMV on device")
{
    // 1 0 0 2 3
    // 2 0 5 0 0
    // 0 4 0 0 1
    unsigned num_rows = 3, num_cols = 5;
    std::vector<int> rows = {0,3,5,7}, cols = {0,3,4,0,2,1,4};
    std::vector<double> vals = {1,2,3,2,5,4,1};
    dg::SparseMatrix<int,double,thrust::host_vector> A ( num_rows, num_cols, rows, cols, vals);
    dg::SparseMatrix<int,double, thrust::device_vector> dA( A);
    SECTION( "transpose")
    {
        // 1 2 0
        // 0 0 4
        // 0 5 0
        // 2 0 0
        // 3 0 1
        auto B = A.transpose();
        CHECK( B.num_rows() == 5);
        CHECK( B.num_cols() == 3);
        CHECK( B.row_offsets() == std::vector<int>{ 0,2,3,4,5,7});
        CHECK( B.column_indices() == std::vector<int>{ 0,1,2,1,0,0,2});
        CHECK( B.values() == std::vector<double>{ 1,2,4,5,2,3,1});
    }
    SECTION( "gemv")
    {
        thrust::host_vector<double> v(5,2), w(3,0);
        for( unsigned i=0; i<v.size(); i++)
            v[i] = double(i+1);
        thrust::device_vector<double> dv( v), dw( w);
        dw = dA*dv;
        // bring to host
        w = dw;
        CHECK( w[0] == 24);
        CHECK( w[1] == 17);
        CHECK( w[2] == 13);
    }
}
