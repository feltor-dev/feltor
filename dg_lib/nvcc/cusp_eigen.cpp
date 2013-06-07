//implementation of SimplicialCholeskyhandler
#include <Eigen/Sparse>
#include <vector>

#include "cusp_eigen.h"
#include "../../lib/timer.h"

Eigen::SparseMatrix<double, Eigen::RowMajor, int> convert( const cusp::coo_matrix<int, double, cusp::host_memory>& cm)
{
    toefl::Timer t;
    typedef Eigen::Triplet<double> T;
    //first generate a vector of Eigen Triplets
    //thrust::host_vector<T> triplets( cm.num_entries);
    t.tic();
    std::vector<T> triplets( cm.num_entries);
    for( unsigned i=0; i<cm.num_entries; i++)
        triplets[i] = T( cm.row_indices[i], cm.column_indices[i], cm.values[i]);
    //now construct the Eigen matrix from triplets (will even sort and reduce triplets)
    Eigen::SparseMatrix<double, Eigen::RowMajor, int> em( cm.num_rows, cm.num_cols);
    em.setFromTriplets( triplets.begin(), triplets.end());
    t.toc();
    std::cout << "Conversion took "<<t.diff()<<"s\n";
    return em;
}


namespace dg
{

struct Impl
{
    typedef typename Eigen::SparseMatrix<double> EMatrix;
    typedef typename Eigen::SimplicialLDLT<EMatrix> SimplicialCholeskyImpl;
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


/*
//unfortunately is not faster
struct PolImpl
{
    typedef typename Eigen::SparseMatrix<double> EMatrix;
    toefl::Timer t;
    PolImpl( const EMatrix& dx, const EMatrix& dy, const EMatrix& jump): dx(dx), dy(dy), jump(jump) {
    dxT = dx.transpose(); 
    dyT = dy.transpose();}
    EMatrix compute( const EMatrix& chi)
    {
        EMatrix laplacian = dxT*chi*dx;// + dy.transpose()*chi*dy + jump;
        return laplacian;
    }
  private:
    EMatrix dx, dy, jump, dxT, dyT;

};
EigenPol::EigenPol(){ pImpl = NULL;}

void EigenPol::construct(const HMatrix& dx, const HMatrix& dy, const HMatrix& jump)
{
    pImpl = new dg::PolImpl( convert(dx), convert(dy), convert(jump)); 
}

EigenPol::~EigenPol(){ delete pImpl;}

void EigenPol::compute( const HMatrix& chi, HMatrix& pol) 
{
    pImpl->t.tic();
    pImpl->compute( convert( chi));
    pImpl->t.toc();
    std::cout << "Computation took " <<pImpl->t.diff()<<"s\n";
}
*/

} //namespace dg



