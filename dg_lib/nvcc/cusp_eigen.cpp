//implementation of SimplicialCholeskyhandler
#include <Eigen/Sparse>
#include <vector>

#include "cusp_eigen.h"
#include "../../lib/timer.h"

Eigen::SparseMatrix<double, Eigen::RowMajor, int> convert( const cusp::coo_matrix<int, double, cusp::host_memory>& cm)
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


namespace dg
{

struct Impl
{
    typedef Eigen::SparseMatrix<double> EMatrix;
    typedef Eigen::SimplicialLDLT<EMatrix> SimplicialCholeskyImpl;
    SimplicialCholeskyImpl solver;
    toefl::Timer t;
    Impl(): solver(){}
    Impl( const EMatrix& matrix): solver(matrix) {}

};

SimplicialCholesky::SimplicialCholesky(): pImpl( new dg::Impl) {}
SimplicialCholesky::SimplicialCholesky(const HMatrix& matrix): 
    pImpl( new dg::Impl( convert(matrix))) {}

SimplicialCholesky::~SimplicialCholesky(){ delete pImpl;}

bool SimplicialCholesky::compute( const HMatrix& matrix) 
{
    pImpl->t.tic();
    pImpl->solver.compute( convert( matrix));
    if( pImpl->solver.info() != Eigen::Success ) return false;
    pImpl->t.toc();
    std::cout << "Decomposition took " <<pImpl->t.diff()<<"s\n";
    return true;
}
bool SimplicialCholesky::solve( double *x, const double* b, unsigned N) 
{
    if( x == b)
    {
        Eigen::Map< Eigen::VectorXd> xmap( x, N);
        xmap = ( (pImpl->solver).solve( xmap));

    }
    else
    {
        Eigen::Map< Eigen::VectorXd> xmap( x, N);
        Eigen::Map< const Eigen::VectorXd> bmap( b, N);
        xmap = (pImpl->solver).solve( bmap);
    }
    if( pImpl->solver.info() != Eigen::Success) return false;
    return true;
}
} //namespace dg



