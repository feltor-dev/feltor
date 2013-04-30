#ifndef _DG_CUSP_EIGEN_H_
#define _DG_CUSP_EIGEN_H_

#include <Eigen/Sparse>
#include <cusp/coo_matrix.h>
#include <vector>
//#include <thrust/host_vector.h>


namespace dg{
typedef Eigen::Triplet<double> T;

Eigen::SparseMatrix<double, Eigen::RowMajor, int> convert( cusp::coo_matrix<int, double, cusp::host_memory>& cm)
{
    //first generate a vector of Eigen Triplets
    //thrust::host_vector<T> triplets( cm.num_entries);
    std::vector<T> triplets( cm.num_entries);
    for( unsigned i=0; i<cm.num_entries; i++)
        triplets[i] = T( cm.row_indices[i], cm.column_indices[i], cm.values[i]);
    //now construct the Eigen matrix from triplets (will even sort and reduce triplets)
    Eigen::SparseMatrix<double, Eigen::RowMajor, int> em( cm.num_rows, cm.num_cols);
    em.setFromTriplets( triplets.begin(), triplets.end());
}

}

#endif // _DG_CUSP_EIGEN_H_
