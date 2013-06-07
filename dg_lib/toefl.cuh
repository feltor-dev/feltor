#ifndef _DG_TOEFL_CUH
#define _DG_TOEFL_CUH


#include "xspacelib.cuh"
#include "blueprint.h"
#include "cg.cuh"

namespace dg
{


template< class T, size_t n, class container=thrust::device_vector<T> >
struct Toefl
{
    typedef std::vector<container> Vector;
    typedef typename thrust::iterator_space<typename container::iterator>::type MemorySpace;
    typedef cusp::ell_matrix<int, T, MemorySpace> Matrix;

    Toefl( const Blueprint<T,n>& bp);

    void exp( const std::vector<container>& y, std::vector<container>& target);
    void log( const std::vector<container>& y, std::vector<container>& target);
    const container& polarisation( const std::vector<container>& y);
    const container& polarisation( ) const { return phi;}
    const Matrix& laplacianM( ) const { return laplaceM;}
    void operator()( const std::vector<container>& y, std::vector<container>& yp);
  private:
    container phi, phi_old;
    container omega, dyphi, chi;
    std::vector<container> expy, dxy, dyy, lapy;

    Matrix A; //contains unnormalized laplacian if local
    Matrix laplaceM; //contains normalized laplacian
    Gamma< Matrix, W2D<T,n> > gamma1;
    ArakawaX<T, n, container> arakawa; 
    Polarisation2dX<T, n, thrust::host_vector<T> > pol; //note the host vector
    CG<container > pcg;
    const Blueprint bp;
    const W2D<T,n> w2d;
    const V2D<T,n> v2d;

};

template< class T, size_t n, class container>
Toefl<T, n, container>::Toefl( const Blueprint& bp): 
    phi( bp.grid().size(), 0.), phi_old(phi), omega( phi), dyphi( phi), chi(phi),
    expy( 2, omega), dxy( expy), dyy( dxy), lapy( dyy),
    gamma1(  laplaceM, w2d, bp.physical().tau[0], bp.physical().a[0]);
    arakawa( bp.grid()), 
    pol(     bp.grid()), 
    pcg( omega, omega.size()), 
    bp( bp), w2d( bp.grid()), v2d( bp.grid())
{
    bp.consistencyCheck();
    //create derivatives
    laplaceM = create::laplacianM( g, normed);
    if( !bp.isEnabled( toefl::TL_GLOBAL) 
        A = create::laplacianM( g, not_normed);

}

//how to set up a computation?
template< class T, size_t n, class container>
const container& Toefl<T, n, container>::polarisation( const std::vector<container>& y)
{
    //compute omega
    if( bp.isEnabled( toefl::TL_GLOBAL))
    {
        exp( y, expy);
        blas1::axpby( -1., expy[0], 1., expy[1], omega); //omega = n_i - n_e
        //compute chi
        blas1::axpby( 1., expy[1], 0., chi);
    }
    else
    {
        blas1::axpby( -1, y[0], 1., y[1], omega);
    }
    //compute S omega 
    blas2::symv( w2d, omega, omega);
    cudaThreadSynchronize();
    if( bp.isEnabled( toefl::TL_GLOBAL))
    {
        cusp::csr_matrix<int, double, MemorySpace> B = pol.create(chi); //first transport to device
        A = B; 
    }
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

#endif //_DG_TOEFL_CUH
