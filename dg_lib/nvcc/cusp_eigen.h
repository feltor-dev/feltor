#ifndef _DG_CUSP_EIGEN_H_
#define _DG_CUSP_EIGEN_H_

#include <cusp/coo_matrix.h>
#include <boost/shared_ptr.hpp>

//USE OF PIMPL IDIOM
//The .h files actually may not include any Eigen library since 
//on the one hand declarations are needed by nvcc and on the other Eigen headers 
//cannot be parsed by nvcc
namespace dg{

struct SimplicialCholesky
{
    typedef cusp::coo_matrix<int, double, cusp::host_memory> HMatrix;
    SimplicialCholesky();
    SimplicialCholesky( const HMatrix& matrix);
    bool compute( const HMatrix& matrix);
    bool solve( double* x, const double* b, unsigned N);
  private:
    struct Impl;
    boost::shared_ptr<Impl > pImpl;
};

}

#endif // _DG_CUSP_EIGEN_H_
