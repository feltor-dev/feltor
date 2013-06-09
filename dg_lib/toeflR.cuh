#ifndef _DG_TOEFLR_CUH
#define _DG_TOEFLR_CUH


#include "xspacelib.cuh"
#include "blueprint.h"
#include "cg.cuh"

namespace dg
{


template< class T, size_t n, class container=thrust::device_vector<T> >
struct ToeflR
{
    typedef std::vector<container> Vector;
    typedef typename thrust::iterator_space<typename container::iterator>::type MemorySpace;
    typedef cusp::ell_matrix<int, T, MemorySpace> Matrix;

    //toeflR is always global
    ToeflR( const Grid<T,n>& g, const Physical& p, const Algorithmic& alg);

    void exp( const std::vector<container>& y, std::vector<container>& target);
    void log( const std::vector<container>& y, std::vector<container>& target);
    const container& polarisation( const std::vector<container>& y);
    const container& polarisation( ) const { return phi;}
    const Matrix& laplacianM( ) const { return laplaceM;}
    void operator()( const std::vector<container>& y, std::vector<container>& yp);
  private:
    container phi, phi_old;
    container n, gamma_n, gamma_phi;
    container omega, dyphi, chi;
    std::vector<container> expy, dxy, dyy, lapy;

    Matrix A; //contains unnormalized laplacian if local
    Matrix laplaceM; //contains normalized laplacian
    Gamma< Matrix, W2D<T,n> > gamma1;
    ArakawaX<T, n, container> arakawa; 
    Polarisation2dX<T, n, thrust::host_vector<T> > pol; //note the host vector
    CG<container > pcg;
    const Physical p;
    const Algorithmic alg;

    const W2D<T,n> w2d;
    const V2D<T,n> v2d;

};

template< class T, size_t n, class container>
ToeflR<T, n, container>::ToeflR( const Grid<T,n>& grid, const Physical& p, const Algorithmic& alg): 
    phi( grid.size(), 0.), phi_old(phi), omega( phi), dyphi( phi), chi(phi),
    expy( p.s.size()+1, omega), dxy( expy), dyy( dxy), lapy( dyy),
    gamma_phi( p.s.size(), phi), gamma_n( gamma_phi),
    gamma1(  laplaceM, w2d, 0, 0);
    arakawa( grid), 
    pol(     grid), 
    pcg( omega, omega.size()), 
    p( p), alg( alg), global( global), w2d( grid), v2d( grid)
{
    //create derivatives
    laplaceM = create::laplacianM( g, normed);
}

template< class T, size_t n, class container>
template< class Matrix> 
unsigned Toefl<T, n, container>::average( const Matrix& m, const container& rhs, container& old, container& sol, double eps)
{
    gamma1.set_species( tau, mu);
    dg::blas2::symv( w2d, y, gamma_y);
    unsigned number = cg( gamma1, gamma_y, y, v2d, eps);

    //extrapolate phi
    blas1::axpby( 2., sol, -1.,  old);
    thrust::swap( sol, old);
    unsigned number = pcg( m, sol , rhs, v2d, eps);

}


//how to set up a computation?
template< class T, size_t n, class container>
const container& Toefl<T, n, container>::polarisation( const std::vector<container>& y)
{
    //compute omega
    exp( y, expy);
    blas2::symv( w2d, expy[1], expy[1]);
    gamma1.set_species( p.s[0].tau, p.s[0].mu);
    pcg( gamma
    blas1::axpby( -1., expy[0], 1., expy[1], omega); //omega = n_i - n_e
    //compute chi
    blas1::axpby( 1., expy[1], 0., chi);
    //compute S omega 
    blas2::symv( w2d, omega, omega);
    cudaThreadSynchronize();
    cusp::csr_matrix<int, double, MemorySpace> B = pol.create(chi); //first transport to device
    A = B; 
    //extrapolate phi
    blas1::axpby( 2., phi, -1.,  phi_old);
    thrust::swap( phi, phi_old);
    unsigned number = pcg( A, phi, omega, v2d, eps);
#ifdef DG_DEBUG
    std::cout << "Number of pcg iterations "<< number <<std::endl;
#endif //DG_DEBUG
    return phi;
}

template< class T, size_t n, class container>
void Toefl<T, n, container>::operator()( const std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 2);
    assert( y.size() == yp.size());

    phi = polarisation( y);

    for( unsigned i=0; i<y.size(); i++)
        arakawa( y[i], phi, yp[i]);

    //compute derivatives
    cudaThreadSynchronize();
    blas2::gemv( arakawa.dy(), phi, dyphi);
    for( unsigned i=0; i<y.size(); i++)
    {
        blas2::gemv( arakawa.dx(), y[i], dxy[i]);
        blas2::gemv( arakawa.dy(), y[i], dyy[i]);
    }
    // curvature terms
    cudaThreadSynchronize();
    blas1::axpby( kappa, dyphi, 1., yp[0]);
    blas1::axpby( kappa, dyphi, 1., yp[1]);
    cudaThreadSynchronize();
    blas1::axpby( -kappa, dyy[0], 1., yp[0]);
    blas1::axpby( -kappa*p.s[i].tau, dyy[i], 1., yp[i]);

    //add laplacians
    for( unsigned i=0; i<y.size(); i++)
    {
        blas2::gemv( laplaceM, y[i], lapy[i]);
        if( global)
        {
            blas1::pointwiseDot( dxy[i], dxy[i], dxy[i]);
            blas1::pointwiseDot( dyy[i], dyy[i], dyy[i]);
            cudaThreadSynchronize();
            //now sum all 3 terms up 
            blas1::axpby( -1., dyy[i], 1., lapy[i]); //behold the minus
            cudaThreadSynchronize();
            blas1::axpby( -1., dxy[i], 1., lapy[i]); //behold the minus
        }
        cudaThreadSynchronize();
        blas1::axpby( -nu, lapy[i], 1., yp[i]); //rescale 
    }


}

template< class T, size_t n, class container>
void Toefl<T, n, container>::exp( const std::vector<container>& y, std::vector<container>& target)
{
    for( unsigned i=0; i<y.size(); i++)
        thrust::transform( y[i].begin(), y[i].end(), target[i].begin(), dg::EXP<T>());
}
template< class T, size_t n, class container>
void Toefl<T, n, container>::log( const std::vector<container>& y, std::vector<container>& target)
{
    for( unsigned i=0; i<y.size(); i++)
        thrust::transform( y[i].begin(), y[i].end(), target[i].begin(), dg::LN<T>());
}

}//namespace dg

#endif //_DG_TOEFLR_CUH
