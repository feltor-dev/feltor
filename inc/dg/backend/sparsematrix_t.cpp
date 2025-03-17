
#include <iostream>
#include "sparsematrix.h"
#include "catch2/catch_all.hpp"

TEST_CASE( "Construct sparse matrix")
{
    size_t num_rows = 3, num_cols = 5, num_nnz = 6;
    thrust::host_vector<int> rows(num_rows+1), cols(num_nnz);
    thrust::host_vector<double> vals(num_nnz);
    SECTION( "Construct")
    {
        dg::SparseMatrix<int,double,thrust::host_vector> mat ( num_rows, num_cols, rows, cols, vals);
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
