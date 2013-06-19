#ifndef _DG_CUSP_EIGEN_H_
#define _DG_CUSP_EIGEN_H_

#include <cusp/coo_matrix.h>
//#include <boost/shared_ptr.hpp> //nvcc cannot parse boost

//USE OF PIMPL IDIOM
//The .h files actually may not include any Eigen library since 
//on the one hand declarations are needed by nvcc and on the other Eigen headers 
//cannot be parsed by nvcc
namespace dg{

struct Impl;

struct SimplicialCholesky
{
    typedef cusp::coo_matrix<int, double, cusp::host_memory> HMatrix;
    SimplicialCholesky();
    ~SimplicialCholesky();
    SimplicialCholesky( const HMatrix& matrix);
    bool compute( const HMatrix& matrix);
    bool solve( double* x, const double* b, unsigned N);
  private:
    SimplicialCholesky( const SimplicialCholesky&);
    SimplicialCholesky& operator=( const SimplicialCholesky&);
    Impl*  pImpl;
};


/*
 //not faster unfortunately
struct PolImpl;

struct EigenPol
{
    typedef cusp::coo_matrix<int, double, cusp::host_memory> HMatrix;
    EigenPol();
    ~EigenPol();
    void construct( const HMatrix& dx, const HMatrix& dy, const HMatrix& jump);
    void compute( const HMatrix& chi, HMatrix& pol);
  private:
    EigenPol( const EigenPol&);
    EigenPol& operator=( const EigenPol&);
    PolImpl*  pImpl;

};
*/

} //namespace dg

#endif // _DG_CUSP_EIGEN_H_
