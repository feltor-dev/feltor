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
    ToeflR( const Grid<T,n>& g, double kappa, double nu, double eps_pol, double eps_gamma);

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

    const W2D<T,n> w2d;
    const V2D<T,n> v2d;
    double eps_pol, eps_gamma; 
    double kappa, nu, tau;

};

template< class T, size_t n, class container>
ToeflR<T, n, container>::ToeflR( const Grid<T,n>& grid, double kappa, double nu, double tau, double eps_pol, double eps_gamma ): 
    phi( grid.size(), 0.), phi_old(phi), omega( phi), dyphi( phi), chi(phi),
    expy( p.s.size()+1, omega), dxy( expy), dyy( dxy), lapy( dyy),
    gamma1(  laplaceM, w2d, -0.5*tau);
    arakawa( grid), 
    pol(     grid), 
    pcg( omega, omega.size()), 
    p( p), alg( alg), w2d( grid), v2d( grid)
    eps_pol(eps_pol), eps_gamma( eps_gamma), kappa(kappa), nu(nu)
{
    //create derivatives
    laplaceM = create::laplacianM( g, normed);
}

template< class T, size_t n, class container>
template< class Matrix> 
unsigned Toefl<T, n, container>::average( const Matrix& m, const container& rhs, container& old, container& sol, double eps)
{
    dg::blas2::symv( w2d, y, gamma_y);
    unsigned number = cg( gamma1, gamma_y, y, v2d, eps);

    //extrapolate phi
    blas1::axpby( 2., sol, -1.,  old);
    thrust::swap( sol, old);
    unsigned number = pcg( m, sol , rhs, v2d, eps);

}


//how to set up a computation?
template< class T, size_t n, class container>
const container& ToeflR<T, n, container>::polarisation( const std::vector<container>& y)
{
    //compute omega
    exp( y, expy);
    blas2::symv( w2d, expy[1], expy[1]);
    //extrapolate gamma-n
    blas1::axpby( 2., gamma_n, -1.,  gamma_old);
    thrust::swap( gamma_n, gamma_old);
    unsigned number;
    number = pcg( gamma1, gamma_n, expy[1], v2d, eps_gamma);
#ifdef DG_DEBUG
    std::cout << "Number of pcg iterations "<< number <<std::endl;
#endif //DG_DEBUG
    blas1::axpby( -1., expy[0], 1., expy[1], omega); //omega = n_i - n_e
    //compute chi
    blas1::axpby( 1., expy[1], 0., chi);
    //compute S omega 
    blas2::symv( w2d, omega, omega);
    cudaThreadSynchronize();
    cusp::csr_matrix<int, double, MemorySpace> B = pol.create(chi); //first transport to device
    A = B; 
    //extrapolate phi
    for( unsigned i=0; i<phi.size(); i++)
    {
        blas1::axpby( 2., phi[i], -1.,  phi_old[i]);
        thrust::swap( phi[i], phi_old[i]);
    }
    unsigned number = pcg( A, phi[0], omega, v2d, eps_pol);
#ifdef DG_DEBUG
    std::cout << "Number of pcg iterations "<< number <<std::endl;
#endif //DG_DEBUG
    //compute Gamma phi[0]
    return phi;
}

template< class T, size_t n, class container>
void ToeflR<T, n, container>::operator()( const std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 2);
    assert( y.size() == yp.size());

    phi = polarisation( y);

    for( unsigned i=0; i<y.size(); i++)
        arakawa( y[i], phi[i], yp[i]);

    //compute derivatives
    cudaThreadSynchronize();
    for( unsigned i=0; i<y.size(); i++)
    {
        blas2::gemv( arakawa.dx(), y[i], dxy[i]);
        blas2::gemv( arakawa.dy(), y[i], dyy[i]);
        blas2::gemv( arakawa.dy(), phi[i], dyphi[i]);
    }
    // curvature terms
    cudaThreadSynchronize();
    blas1::axpby( kappa, dyphi[0], 1., yp[0]);
    blas1::axpby( kappa, dyphi[1], 1., yp[1]);
    cudaThreadSynchronize();
    blas1::axpby( -1.*kappa, dyy[0], 1., yp[0]);
    blas1::axpby( tau*kappa, dyy[1], 1., yp[1]);

    //add laplacians
    for( unsigned i=0; i<y.size(); i++)
    {
        blas2::gemv( laplaceM, y[i], lapy[i]);
        blas1::pointwiseDot( dxy[i], dxy[i], dxy[i]);
        blas1::pointwiseDot( dyy[i], dyy[i], dyy[i]);
        cudaThreadSynchronize();
        //now sum all 3 terms up 
        blas1::axpby( -1., dyy[i], 1., lapy[i]); //behold the minus
        cudaThreadSynchronize();
        blas1::axpby( -1., dxy[i], 1., lapy[i]); //behold the minus
        cudaThreadSynchronize();
        blas1::axpby( -nu, lapy[i], 1., yp[i]); //rescale 
    }


}

template< class T, size_t n, class container>
void ToeflR<T, n, container>::exp( const std::vector<container>& y, std::vector<container>& target)
{
    for( unsigned i=0; i<y.size(); i++)
        thrust::transform( y[i].begin(), y[i].end(), target[i].begin(), dg::EXP<T>());
}
template< class T, size_t n, class container>
void ToeflR<T, n, container>::log( const std::vector<container>& y, std::vector<container>& target)
{
    for( unsigned i=0; i<y.size(); i++)
        thrust::transform( y[i].begin(), y[i].end(), target[i].begin(), dg::LN<T>());
}

}//namespace dg

#endif //_DG_TOEFLR_CUH
