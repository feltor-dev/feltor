#pragma once

#include <exception>

#include "dg/backend/xspacelib.cuh"
#include "dg/algorithm.h"

#ifdef DG_BENCHMARK
#include "dg/backend/timer.cuh"
#endif


namespace dg
{
template< class container>
struct Diffusion
{
    Diffusion( const dg::Grid2d<double>& g, double nu): nu_(nu),
        w2d(dg::create::weights( g)), v2d( dg::create::inv_weights(g)), temp( g.size()) { 
        LaplacianM = dg::create::laplacianM( g, dg::normed, dg::symmetric); 
        }
    void operator()( const container& x, container& y)
    {
        dg::blas2::gemv( LaplacianM, x, temp);
        dg::blas2::gemv( LaplacianM, temp, y);
        //dg::blas2::gemv( LaplacianM, y, temp);
        //dg::blas1::axpby( 0., y, -nu_ , y);
        dg::blas1::scal( y, -nu_);
    }
    const container& weights(){return w2d;}
    const container& precond(){return v2d;}
  private:
    double nu_;
    const container w2d, v2d;
    container temp;
    dg::DMatrix LaplacianM;
};


template< class container=thrust::device_vector<double> >
struct Mima
{
    typedef std::vector<container> Vector;
    typedef typename container::value_type value_type;
    typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    //typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    typedef dg::DMatrix Matrix; //fastest device Matrix (does this conflict with 

    /**
     * @brief Construct a Mima solver object
     *
     * @param g The grid on which to operate
     * @param kappa The curvature
     * @param nu The artificial viscosity
     * @param tau The ion temperature
     * @param eps_pol stopping criterion for polarisation equation
     * @param eps_gamma stopping criterion for Gamma operator
     * @param global local or global computation
     */
    Mima( const Grid2d<value_type>& g, double kappa, double eps, bool global);

    /**
     * @brief Returns phi and psi that belong to the last y in operator()
     *
     * In a multistep scheme this belongs to the point HEAD-1
     * @return phi[0] is the electron and phi[1] the generalized ion potential
     */
    const container& potential( ) const { return phi;}

    /**
     * @brief Return the normalized negative laplacian used by this object
     *
     * @return cusp matrix
     */
    const Matrix& laplacianM( ) const { return laplaceM;}

    /**
     * @brief Compute the right-hand side of the toefl equations
     *
     * @param y input vector
     * @param yp the rhs yp = f(y)
     */
    void operator()( const container& y, container& yp);


  private:
    double kappa;
    bool global;
    container phi, dxphi, dyphi, omega;
    container dxxphi, dxyphi;

    //matrices and solvers
    Matrix laplaceM; //contains normalized laplacian
    ArakawaX< Matrix, container> arakawa; 
    const container w2d, v2d;
    Invert<container> invert;
    Helmholtz<Matrix, container, container> helmholtz;



};

template< class container>
Mima< container>::Mima( const Grid2d<value_type>& grid, double kappa, double eps, bool global ): 
    kappa( kappa), global(global),
    phi( grid.size(), 0.), dxphi( phi), dyphi( phi), omega(phi),
    dxxphi( phi), dxyphi(phi),
    arakawa( grid), 
    w2d( create::weights(grid)), v2d( create::inv_weights(grid)),
    laplaceM( create::laplacianM( grid, normed, dg::symmetric)),
    helmholtz( grid, -1),
    invert( phi, grid.size(), eps)
{
}

template< class container>
void Mima< container>::operator()( const container& y, container& yp)
{
    invert( helmholtz, phi, y);
    dg::blas1::axpby( 1., phi, -1., y, omega); //omega = lap \phi


    arakawa( phi, omega, yp);
    //compute derivatives
    blas2::gemv( arakawa.dx(), phi, dxphi);
    blas2::gemv( arakawa.dy(), phi, dyphi);
    blas2::gemv( arakawa.dx(), dxphi, dxxphi);
    blas2::gemv( arakawa.dy(), dxphi, dxyphi);
    //gradient terms
    blas1::axpby( -1, dyphi, 1., yp);

    blas1::pointwiseDot( dyphi, omega, omega);
    blas1::axpby( -2*kappa, omega, 1., yp);

    if( global)
    {
        blas1::pointwiseDot( dxphi, dxyphi, omega);
        blas1::axpby( -kappa, omega, 1., yp);
        blas1::pointwiseDot( dyphi, dxxphi, omega);
        blas1::axpby( +kappa, omega, 1., yp);
    }
    //dg::blas1::scal(yp, -1.);


}


}//namespace dg

