#pragma once


#include "parameters.h"

#include "dg/backend/xspacelib.cuh"
#include "dg/cg.h"
#include "dg/algorithm.h"

#ifdef DG_BENCHMARK
#include "dg/backend/timer.cuh"
#endif



namespace eule
{

template<class container>
struct Diffusion
{
    Diffusion( const dg::Grid2d<double>& g, double nu, double mue_hat, double mui_hat):
        nu_(nu), mue_hat(mue_hat), mui_hat(mui_hat), 
        w2d_( dg::create::weights(g)), v2d_( dg::create::inv_weights(g)), 
        temp( g.size()){

        LaplacianM_perp = dg::create::laplacianM( g, dg::normed);

    }
    void operator()( const std::vector<container>& x, std::vector<container>& y)
    {
        for( unsigned i=0; i<x.size(); i++)
        {
            dg::blas2::gemv( LaplacianM_perp, x[i], temp);
            dg::blas2::gemv( LaplacianM_perp, temp, y[i]);
        }
        dg::blas1::scal( y[0], -nu_);
        dg::blas1::scal( y[1], -nu_);
        dg::blas1::scal( y[2], -nu_/mue_hat);
        dg::blas1::scal( y[3], -nu_/mui_hat);

    }
    const dg::DMatrix& laplacianM()const {return LaplacianM_perp;}
    const container& weights(){return w2d_;}
    const container& precond(){return v2d_;}

  private:
    double nu_, mue_hat, mui_hat;
    container w2d_, v2d_;
    container temp;
    dg::DMatrix LaplacianM_perp;
};

template< class container=thrust::device_vector<double> >
struct Asela
{
    typedef typename container::value_type value_type;
    typedef dg::DMatrix Matrix; //fastest device Matrix (does this conflict with 

    /**
     * @brief Construct a Asela solver object
     *
     * @param g The grid on which to operate
     * @param p The parameters
     */
    Asela( const dg::Grid2d<value_type>& g, Parameters p);

    /**
     * @brief Exponentiate pointwise every Vector in src 
     *
     * @param src source
     * @param dst destination may equal source
     */
    void exp( const std::vector<container>& src, std::vector<container>& dst, unsigned howmany);

    /**
     * @brief Take the natural logarithm pointwise of every Vector in src 
     *
     * @param src source
     * @param dst destination may equal source
     */
    void log( const std::vector<container>& src, std::vector<container>& dst, unsigned howmany);

    /**
     * @brief Returns phi and psi that belong to the last y in operator()
     *
     * In a multistep scheme this belongs to the point HEAD-1
     * @return phi[0] is the electron and phi[1] the generalized ion potential
     */
    const std::vector<container>& potential( ) const { return phi;}
    const container& aparallel( ) const { return apar;}

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
    void operator()( std::vector<container>& y, std::vector<container>& yp);

  private:
    const container w2d, v2d, one;
    container rho, omega, apar;

    std::vector<container> phi, arakAN, arakAU, u;
    std::vector<container> expy;

    //matrices and solvers
    Matrix laplaceM; //contains negative normalized laplacian

    dg::ArakawaX<Matrix, container> arakawa; 
    dg::Invert<container> invert_pol, invert_maxwell; 
    dg::Helmholtz<Matrix, container, container> maxwell;

    dg::Elliptic< Matrix, container, container > pol; 

    Parameters p;

};

template< class container>
Asela< container>::Asela( const dg::Grid2d<value_type>& grid, Parameters p ): 
    w2d( dg::create::weights(grid)),
    v2d( dg::create::inv_weights(grid)),
    one( dg::evaluate( dg::one, grid)),
    rho( dg::evaluate( dg::zero, grid)),
    omega(rho), apar(rho),
    phi( 2, rho), expy( phi), arakAN( phi), arakAU( phi), u(phi), 
    laplaceM (dg::create::laplacianM( grid, dg::normed, dg::centered)),
    arakawa( grid), 
    maxwell( grid),
    invert_pol( rho, rho.size(), p.eps_pol),
    invert_maxwell( rho, rho.size(), p.eps_maxwell),
    pol(     grid), 
    p(p)
{ }

template< class container>
void Asela< container>::operator()( std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 4);
    assert( y.size() == yp.size());

    //solve polarisation equation
    exp( y, expy, 2);
    dg::blas1::axpby( p.dhat[1]*p.dhat[1], expy[1], 0., omega);
    pol.set_chi(omega);
    dg::blas1::axpby( -p.dhat[0], expy[0], p.dhat[1], expy[1], rho);
    invert_pol( pol, phi[0], rho, w2d, v2d);
    //compute phi[1]
    arakawa.bracketS( phi[0], phi[0], phi[1]);
    dg::blas1::axpby( 1., phi[0], -0.5*p.dhat[1], phi[1]);////////////////////

    //solve induction equation
    dg::blas1::axpby( -1./p.dhat[1], expy[0], 0., omega);
    dg::blas1::axpby( -1./p.dhat[1], expy[1], 1., omega);
    maxwell.set_chi(omega);
    dg::blas1::pointwiseDot( expy[0], y[2], rho);
    dg::blas1::pointwiseDot( expy[1], y[3], omega);
    dg::blas1::axpby( -1./p.dhat[1],omega , -1./p.dhat[1], rho);
    invert_maxwell( maxwell, apar, rho);
    dg::blas1::axpby( -1./p.dhat[0]/p.dhat[0], y[2], 1./p.dhat[0]/p.dhat[0], apar, u[0]);
    dg::blas1::axpby( 1./p.dhat[1]/p.dhat[1], y[3], -1./p.dhat[1]/p.dhat[1], apar, u[1]);

    double sign[2]={-1.,1.};
    for( unsigned i=0; i<2; i++)
    {
        arakawa( y[i], phi[i], yp[i]);
        arakawa( y[2+i], phi[i], yp[2+i]);
        arakawa( apar, y[i], arakAN[i]);
        arakawa( apar, u[i], arakAU[i]);
        dg::blas1::pointwiseDot( u[i], arakAN[i], rho);
        dg::blas1::pointwiseDot( u[i], arakAU[i], omega);
        dg::blas1::axpby( p.dhat[i], rho, 1., yp[i]);
        dg::blas1::axpby( sign[i]*p.dhat[i]*p.dhat[i]*p.dhat[i], omega, 1., yp[2+i]);
        dg::blas1::axpby( p.dhat[i], arakAU[i], 1., yp[i]);
        dg::blas1::axpby( sign[i]*p.rhohat[i]*p.rhohat[i]/p.dhat[i], arakAN[i], 1., yp[2+i]);
    }

}

template< class container>
void Asela< container>::exp( const std::vector<container>& y, std::vector<container>& target, unsigned howmany)
{
    for( unsigned i=0; i<howmany; i++)
        dg::blas1::transform( y[i], target[i], dg::EXP<value_type>());
}
template< class container>
void Asela< container>::log( const std::vector<container>& y, std::vector<container>& target, unsigned howmany)
{
    for( unsigned i=0; i<howmany; i++)
        dg::blas1::transform( y[i], target[i], dg::LN<value_type>());
}


}//namespace eule

