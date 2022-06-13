#include <iostream>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include <cusp/transpose.h>
#include <cusp/multiply.h>
#include <cusp/elementwise.h>
#include "precond.h"
#include "topology/operator.h"
#include "backend/timer.h"

int main()
{
    std::cout << "Test SAINV preconditioner\n";
    // We generate the 1d Laplacian operator
    unsigned size = 100;
    unsigned nnz = 10;
    double eps = 1e-3;
    std::cout << "Type size of matrix (100), max number nnz per row (10), absolute threshold (1e-3)\n";
    std::cin >> size >> nnz >> eps;
    thrust::host_vector<double> weights( size, 1.), d(weights);
    for( int i=0; i<(int)size; i++)
        weights[i] = (1.+(i%3)/2.)/10.;
    dg::Operator<double> op( size, 0.);
    op( 0,0) = 2/weights[0], op( 0, 1) = -1./weights[0];
    for( int i=1; i<(int)size-1; i++)
    {
        op( i, i+1) = op( i, i-1) = -1./weights[i];
        op(i,i) = 2./weights[i];
    }
    op( size-1, size-1) = 2./weights[size-1], op( size-1, size-2) = -1./weights[size-1];

    // Here compute the inverse
    auto inverse = dg::create::inverse( op);

    // assign to a coo matrix
    using coo_mat = cusp::coo_matrix<int, double, cusp::host_memory>;
    using csr_mat = cusp::csr_matrix<int, double, cusp::host_memory>;

    coo_mat a( size, size, 4+3*(size-2)), ainv(size,size, size*size);
    unsigned counter = 0;
    for( int i=0; i<(int)size; i++)
        for( int k=0; k<(int)size; k++)
        {
            if( op(i,k) != 0)
            {
                a.row_indices[counter] = i;
                a.column_indices[counter] = k;
                a.values[counter] = op(i,k);
                counter ++;
            }
            ainv.row_indices[i*size+k] = i;
            ainv.column_indices[i*size+k] = k;
            ainv.values[i*size+k] = inverse(i,k);
        }
    csr_mat a_csr = a, z;

    //std::cout << "Matrix A\n";
    //cusp::print( a_csr);
    //cusp::print(a_csr.row_offsets);

    std::cout << "Create preconditioner\n";
    dg::Timer t;
    t.tic();
    dg::create::sainv_precond( a_csr, z, d, weights, nnz, eps);
    t.toc();
    std::cout <<"took "<<t.diff()<<"s"<<std::endl;
    //std::cout << "Resulting z \n";
    //cusp::print( z);

    // Test if Z^T D^{-1} Z W = Ainv
    //
    coo_mat zT, zTD;
    cusp::transpose( z, zT);
    // create D^{-1}
    coo_mat dinv( size, size, size);
    coo_mat wei( size, size, size);
    for( unsigned i=0; i<size; i++)
    {
        dinv.row_indices[i] = dinv.column_indices[i] = i;
        wei.row_indices[i] = wei.column_indices[i] = i;
        dinv.values[i] = 1./d[i];
        wei.values[i] = weights[i];
    }
    cusp::multiply( zT, dinv, zTD);
    cusp::multiply( zTD, z, zT);
    cusp::multiply( zT, wei, dinv);
    dinv.sort_by_row_and_column();
    //std::cout << "Numerical inverse:\n";
    //cusp::print( zT);
    //std::cout << "Actual inverse:\n";
    //cusp::print( ainv);
    // Compute matrix norm between difference
    cusp::subtract( ainv, dinv, zTD);
    double norm_err = 0.0;
    double norm_ainv = 0.0;
    for( unsigned u=0; u<zTD.values.size(); u++)
        norm_err += zTD.values[u]*zTD.values[u];
    for( unsigned u=0; u<ainv.values.size(); u++)
        norm_ainv += ainv.values[u]*ainv.values[u];

    std::cout << "Number of nonzeros per line in Z "<<z.values.size()/(double)size<<"\n";
    std::cout << "Absolute error norm (not small) "<<sqrt(norm_err)<<"\n";
    std::cout << "Relative error norm (not small) "<<sqrt(norm_err/norm_ainv)<<"\n";


    return 0;
}

