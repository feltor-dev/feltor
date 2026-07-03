
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "tensor_traits_std.h"
#include "tensor_traits_thrust.h"
#include "sparsematrix.h"
#include "../blas1.h"
#include "../blas2.h"
#include "catch2/catch_all.hpp"

// MW Possible update: With the newest thrust version std::vector can be replaced by thrust::host_vector everywhere
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
    thrust::device_vector<int> rows(num_rows+1), cols(num_nnz);
    thrust::device_vector<double> vals(num_nnz);
    SECTION( "Construct")
    {
        dg::SparseMatrix<int,double,thrust::device_vector> mat ( num_rows, num_cols, rows, cols, vals);
        static_assert( std::is_same_v< dg::get_tensor_category< dg::SparseMatrix<
            int,double,thrust::device_vector>>, dg::SparseMatrixTag>);
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
        mat.num_rows() = 7;
        CHECK( mat.num_rows() == 7);
        mat.num_cols() = 12;
        CHECK( mat.num_cols() == 12);
        mat.row_offsets().resize( 8);
        mat.column_indices().resize( 9);
        mat.values().resize( 9);
        CHECK( mat.num_nnz() == 9);
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
        A.sort_indices();
        CHECK( A.row_offsets() == std::vector{0,3,5,7});
        CHECK( A.column_indices() == std::vector{0,3,4,0,2,1,4});
        CHECK( A.values() == std::vector<double>{1,2,3,2,5,4,1});
    }
    SECTION( "setFromCoo")
    {
        //![setFromCoo]
        // 1 0 0 2 3
        // 2 0 5 0 0
        // 0 4 0 0 1
        unsigned num_rows = 3, num_cols = 5;
        std::vector<int> rows = {2,2,0,0,0,1,1}, cols = {4,1,4,3,0,2,0};
        std::vector<double> vals = {1,4, 3,2,1,5,2};
        dg::SparseMatrix<int,double,std::vector> A;
        A.setFromCoo( num_rows, num_cols, rows, cols, vals, true);
        CHECK( A.row_offsets() == std::vector{0,3,5,7});
        CHECK( A.column_indices() == std::vector{0,3,4,0,2,1,4});
        CHECK( A.values() == std::vector<double>{1,2,3,2,5,4,1});
        //![setFromCoo]
    }

}
TEST_CASE( "Linear algebra")
{
    //![csr_ctor]
    // 1 0 0 2 3
    // 2 0 5 0 0
    // 0 4 0 0 1
    unsigned num_rows = 3, num_cols = 5;
    std::vector<int> rows = {0,3,5,7}, cols = {0,3,4,0,2,1,4};
    std::vector<double> vals = {1,2,3,2,5,4,1};
    dg::SparseMatrix<int,double,std::vector> A ( num_rows, num_cols, rows, cols, vals);
    //
    CHECK( A.num_rows() == num_rows);
    CHECK( A.num_cols() == num_cols);
    CHECK( A.num_vals() == vals.size());
    CHECK( A.row_offsets()    == rows);
    CHECK( A.column_indices() == cols);
    CHECK( A.values()         == vals);
    //![csr_ctor]

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
        //![set]
        // 0 2 0 0 4
        // 0 1 2 0 0
        // 0 0 0 0 0
        num_rows = 3, num_cols = 5;
        rows = {0,2,4,4}, cols = {1,4,1,2};
        vals = {2,4,1,2};
        dg::SparseMatrix<int,double,std::vector> C;
        C.set( num_rows, num_cols, rows, cols, vals);
        //![set]

        // 1 2 0 2 7
        // 2 1 7 0 0
        // 0 4 0 0 1
        C += A; // The way this is implemented this will also test C + A
        CHECK( C.num_rows() == 3);
        CHECK( C.num_cols() == 5);
        CHECK( C.row_offsets() == std::vector<int>{ 0,4,7,9});
        CHECK( C.column_indices() == std::vector<int>{ 0,1,3,4,0,1,2,1,4});
        CHECK( C.values() == std::vector<double>{ 1,2,2,7,2,1,7,4,1});
        C -= A;
        CHECK( C.num_rows() == 3);
        CHECK( C.num_cols() == 5);
        CHECK( C.row_offsets() == std::vector<int>{ 0,4,7,9});
        CHECK( C.column_indices() == std::vector<int>{ 0,1,3,4,0,1,2,1,4});
        CHECK( C.values() == std::vector<double>{ 0,2,0,4,0,1,2,0,0});

    }
    SECTION( "Scal")
    {

        // 0.5 0 0 1 1.5
        // 1 0 2.5 0 0
        // 0 2 0 0 0.5
        auto C = -A;
        C = 0.5*C;
        CHECK( C.num_rows() == 3);
        CHECK( C.num_cols() == 5);
        CHECK( C.row_offsets() == std::vector<int>{ 0,3,5,7});
        CHECK( C.column_indices() == std::vector<int>{ 0,3,4,0,2,1,4});
        CHECK( C.values() == std::vector<double>{ -0.5,-1,-1.5,-1,-2.5,-2,-0.5});
    }
}
TEST_CASE( "SpMV on device")
{
    //![host2device]
    // 1 0 0 2 3
    // 2 0 5 0 0
    // 0 4 0 0 1
    unsigned num_rows = 3, num_cols = 5;
    // unsorted!!
    std::vector<int> rows = {0,3,5,7}, cols = {0,4,3,0,2,1,4};
    std::vector<double> vals = {1,3,2,2,5,4,1};
    dg::SparseMatrix<int,double,thrust::host_vector> A ( num_rows, num_cols, rows, cols, vals);
    dg::SparseMatrix<int,double, thrust::device_vector> dA( A);
    //![host2device]
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

TEST_CASE("Documentation")
{
    // redirect std::cout to string stream
    std::stringstream ss;
    auto cout_buf = std::cout.rdbuf( ss.rdbuf());
    //![summary]
    // 1 0 0 2
    // 2 0 5 0
    // 0 4 0 0
    // 0 1 1 0
    unsigned num_rows = 4, num_cols = 4;
    std::vector<int> rows = {0,2,4,5,7}, cols = {0,3,0,2,1,1,2};
    std::vector<double> vals = {1,2,2,5,4,1,1};
    dg::SparseMatrix<int,double,std::vector> A ( num_rows, num_cols, rows, cols, vals);
    dg::print( A);
    //
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
    dg::print( B);
    //
    // 3 4 0 0
    // 10 16.5 13.5 8.5
    // 0 20 2 4
    // 10 0 0 2.5
    // 5.5 2 0 1
    auto C = B*A.transpose()+0.5*B;
    dg::print( C);
    //
    CHECK( C.num_rows() == 5);
    CHECK( C.num_cols() == 4);
    CHECK( C.row_offsets() == std::vector<int>{ 0,2,6,9,11,14});
    CHECK( C.column_indices() == std::vector<int>{ 0,1,0,1,2,3,1,2,3,0,3,0,1,3});
    CHECK( C.values() == std::vector<double>{ 3, 4, 10, 16.5, 13.5, 8.5, 20, 2, 4, 10, 2.5, 5.5, 2, 1});
    //
    // Matrix-vector multiplication can be done on device
    dg::SparseMatrix<int,double,thrust::device_vector> dC = C;
    std::vector<double>  v = { 2,4,6,8}, w = {1,2,3,4,5};
    thrust::device_vector<double> dv( v.begin(), v.end()), dw( w.begin(), w.end());
    // dw = dC * dv + 0.5 dw
    dg::blas2::gemv( 1., dC, dv, 0.5, dw);
    //
    thrust::copy( dw.begin(), dw.end(), w.begin());
    CHECK( w == std::vector{22.5,236.,125.5,42.,29.5});
    //![summary]
    // reset buffer
    std::cout.rdbuf( cout_buf);
}
