#ifndef _DG_SHU_CUH
#define _DG_SHU_CUH

#include <cusp/ell_matrix.h>

#include "dg/blas.h"
#include "dg/dlt.h"
//#include "dg/cusp_eigen.h"
#include "dg/arakawa.cuh"
#include "dg/derivatives.cuh"
#include "dg/cg.cuh"

namespace dg
{

template< class container=thrust::device_vector<double> >
struct Shu 
{
    typedef typename container::value_type value_type;
    typedef container Vector;
    typedef typename thrust::iterator_space<typename container::iterator>::type MemorySpace;
    typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;

    Shu( const Grid<value_type>& grid, double D, double eps = 1e-4);

    Matrix& lap() { return laplace;}
    const container& potential( ) {return phi;}
    void operator()( const Vector& y, Vector& yp);
  private:
    //typedef typename VectorTraits< Vector>::value_type value_type;
    Matrix laplace;
    container omega, phi, phi_old;
    Arakawa< container> arakawa; 
    CG< container > pcg;
    //SimplicialCholesky cholesky;
    thrust::host_vector<double> x,b;
    T2D<value_type> t2d;
    S2D<value_type> s2d;
    
    double D;
    double eps; 
};

template< class container>
Shu< container>::Shu( const Grid<value_type>& g, double D, double eps): 
    omega( g.size(), 0.), phi(omega), phi_old(phi),
    arakawa( g), 
    pcg( omega, g.size()), x( phi), b( x),
    t2d( g), s2d( g), D(D), eps(eps)
{
    typedef cusp::coo_matrix<int, value_type, MemorySpace> HMatrix;
    HMatrix A = dg::create::laplacianM( g, not_normed, LSPACE);
    laplace = A;
    //cholesky.compute( A);
}

template< class container>
void Shu<container>::operator()( const Vector& y, Vector& yp)
{
    dg::blas2::symv( laplace, y, yp);
    dg::blas2::symv( -D, t2d, yp, 0., yp); //laplace is unnormalized -laplace
    //compute S omega
    blas2::symv( s2d, y, omega);
    blas1::axpby( 2., phi, -1.,  phi_old);
    phi.swap( phi_old);
    unsigned number = pcg( laplace, phi, omega, t2d, eps);
    //std::cout << "Number of pcg iterations "<< number<<"\n"; 
    //b = omega; //copy data to host
    //cholesky.solve( x.data(), b.data(), b.size()); //solve on host
    //phi = x; //copy data back to device
    arakawa( y, phi, omega); //A(y,phi)-> omega
    blas1::axpby( 1., omega, 1., yp);

}

}//namespace dg

#endif //_DG_SHU_CUH
