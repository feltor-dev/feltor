#ifndef _DG_CUSP_EIGEN_H_
#define _DG_CUSP_EIGEN_H_

#include <Eigen/Sparse>
#include <cusp/coo_matrix.h>
#include <vector>
//#include <thrust/host_vector.h>

//The .h files actually may not include any Eigen library since 
//on the one hand declarations are needed by nvcc and Eigen headers 
//cannot be parsed by nvcc
namespace dg{

Eigen::SparseMatrix<double, Eigen::RowMajor, int> convert( cusp::coo_matrix<int, double, cusp::host_memory>& cm)
{
    typedef Eigen::Triplet<double> T;
    //first generate a vector of Eigen Triplets
    //thrust::host_vector<T> triplets( cm.num_entries);
    std::vector<T> triplets( cm.num_entries);
    for( unsigned i=0; i<cm.num_entries; i++)
        triplets[i] = T( cm.row_indices[i], cm.column_indices[i], cm.values[i]);
    //now construct the Eigen matrix from triplets (will even sort and reduce triplets)
    Eigen::SparseMatrix<double, Eigen::RowMajor, int> em( cm.num_rows, cm.num_cols);
    em.setFromTriplets( triplets.begin(), triplets.end());
    return em;
}

typedef Eigen::SimplicialCholesky< Eigen::SparseMatrix<double> > SimplicialCholesky;

}

#endif // _DG_CUSP_EIGEN_H_
