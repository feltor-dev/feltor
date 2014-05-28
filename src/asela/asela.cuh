#pragma once

#include "dg/xspacelib.cuh"
#include "dg/cg.cuh"
#include "dg/gamma.cuh"

#ifdef DG_BENCHMARK
#include "dg/timer.cuh"
#endif



namespace eule
{

template<class container>
struct Diffusion
{
    Diffusion( const dg::Grid2d<double>& g, double nu, double mue_hat, double mui_hat):nu_(nu), mue_hat(mue_hat), mui_hat(mui_hat), w2d( 4, dg::create::w2d(g)), v2d( 4, dg::create::v2d(g)), temp( g.size()){
        LaplacianM_perp = dg::create::laplacianM( g, dg::normed, dg::XSPACE);
    }
    void operator()( const std::vector<container>& x, std::vector<container>& y)
    {
        for( unsigned i=0; i<x.size(); i++)
        {
            dg::blas2::gemv( LaplacianM_perp, x[i], temp);
            dg::blas2::gemv( LaplacianM_perp, temp, y[i]);
        }
        dg::blas1::axpby( -nu_, y[0], 0., y[0]);
        dg::blas1::axpby( -nu_, y[1], 0., y[1]);
        dg::blas1::axpby( -nu_/mue_hat, y[2], 0., y[2]);
        dg::blas1::axpby( -nu_/mui_hat, y[3], 0., y[3]);

    }
    const dg::DMatrix& laplacianM()const {return LaplacianM_perp;}
    const std::vector<container>& weights(){return w2d;}
    const std::vector<container>& precond(){return v2d;}

  private:
    double nu_, mue_hat, mui_hat;
    const std::vector<container> w2d, v2d;
    container temp;
    dg::DMatrix LaplacianM_perp;
};

template< class container=thrust::device_vector<double> >
struct Asela
{
    //typedef std::vector<container> Vector;
    typedef typename container::value_type value_type;
    //typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    //typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    typedef dg::DMatrix Matrix; //fastest device Matrix (does this conflict with 
    //typedef in ArakawaX ??

    /**
     * @brief Construct a Asela solver object
     *
     * @param g The grid on which to operate
     * @param p The parameters
     */
    Asela( const Grid2d<value_type>& g, Parameters p);

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

    /**
     * @brief Return the normalized negative laplacian used by this object
     *
     * @return cusp matrix
     */
    const Matrix& laplacianM( ) const { return laplaceM;}

    /**
     * @brief Return the Gamma operator used by this object
     *
     * @return Gamma operator
     */
    const Gamma<Matrix, container >&  gamma() const {return gamma1;}

    /**
     * @brief Compute the right-hand side of the toefl equations
     *
     * @param y input vector
     * @param yp the rhs yp = f(y)
     */
    void operator()( const std::vector<container>& y, std::vector<container>& yp);

    /**
     * @brief Return the mass of the last field in operator() in a global computation
     *
     * @return int exp(y[0]) dA
     * @note undefined for a local computation
     */
    double mass( ) {return mass_;}
    /**
     * @brief Return the last integrated mass diffusion of operator() in a global computation
     *
     * @return int \nu \Delta (exp(y[0])-1)
     * @note undefined for a local computation
     */
    double mass_diffusion( ) {return diff_;}
    /**
     * @brief Return the energy of the last field in operator() in a global computation
     *
     * @return integrated total energy in {ne, ni}
     * @note undefined for a local computation
     */
    double energy( ) {return energy_;}
    /**
     * @brief Return the integrated energy diffusion of the last field in operator() in a global computation
     *
     * @return integrated total energy diffusion
     * @note undefined for a local computation
     */
    double energy_diffusion( ){ return ediff_;}

  private:
    const container w2d, v2d, one;
    container rho, omega;

    std::vector<container> phi;
    std::vector<container> expy;

    //matrices and solvers
    Matrix A; //contains polarisation matrix
    Matrix laplaceM; //contains negative normalized laplacian
    dg::ArakawaX< container> arakawa; 
    dg::Invert<Matrix, container> invert_A, invert_maxwell; 
    dg::Maxwell<Matrix, container> maxwell;
    dg::Polarisation2dX< thrust::host_vector<value_type> > pol; //note the host vector


    Parameters p;
    double mass_, energy_, diff_, ediff_;

};

template< class container>
Asela< container>::Asela( const Grid2d<value_type>& grid, Parameters p ): 
    w2d( create::w2d(grid)), v2d( create::v2d(grid)), one( grid.size(), 1.),
    rho( grid.size(), 0.), omega(chi)
    phi( 2, chi), expy( phi), 
    arakawa( grid), 
    maxwell( laplaceM, w2d, v2d),
    invert_A( chi, chi.size(), p.eps_pol),
    invert_maxwell( chi, chi.size(), p.eps_maxwell),
    pol(     grid), 
    p(p)
{
    //create derivatives
    laplaceM = create::laplacianM( grid, normed, dg::XSPACE, dg::symmetric); //doesn't hurt to be symmetric but doesn't solver pb
    A = create::laplacianM( grid, not_normed, dg::XSPACE, dg::symmetric);

}

template< class container>
void Asela< container>::operator()( const std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 2);
    assert( y.size() == yp.size());

    //solve polarisation equation
    exp( y, expy, 2);
    A = pol.create( expy[1]);
    dg::blas1::axpby( -1., expy[0], 1., expy[1], rho);
    invert_A( A, phi[0], rho, w2d, v2d);
    //compute phi[1]
    arakawa.bracket( phi[0], phi[0], phi[1]);
    blas1::axpby( 1., phi[0], -0.5, phi[1]);

    //solve induction equation
    dg::blas1::axpby( p.beta/p.mu[0], expy[0], 0., maxwell.chi());
    dg::blas1::axpby( -p.beta/p.mu[1], expy[1], 1., maxwell.chi());
    dg::blas1::pointwiseDot( expy[0], y[2], rho);
    dg::blas1::pointwiseDot( expy[1], y[3], omega);
    dg::blas1::axpby( -1.,omega , 1., rho);
    invert_maxwell( maxwell, apar, rho);
    dg::blas1::axpby( 1., y[2], -p.beta/p.mu[0], apar, u[0]);
    dg::blas1::axpby( 1., y[3], -p.beta/p.mu[1], apar, u[1]);

    for( unsigned i=0; i<y.size(); i++)
        arakawa( y[i], phi[i], yp[i]);
    for( unsigned i=0; i<2; i++)
    {
        arakawa( apar, y[i], arakAN[i]);
        arakawa( apar, u[i], arakAU[i]);
        dg::pointwiseDot( u[i], arakAN[i], rho);
        dg::pointwiseDot( u[i], arakAU[i], omega);
        dg::blas1::axpby( p.beta*eps_hat, rho, 1., yp[i]);
        dg::blas1::axpby( p.beta*eps_hat, omega, 1., yp[2+i]);
        dg::blas1::axpby( p.beta, arakAU[i], 1., yp[i]);
        dg::blas1::axpby( p.beta/p.mu[i]*p.tau[i], arakAN[i], 1., yp[2+i]);
    }

}

template< class container>
void Asela< container>::exp( const std::vector<container>& y, std::vector<container>& target, unsigned howmany)
{
    for( unsigned i=0; i<howmany; i++)
        thrust::transform( y[i].begin(), y[i].end(), target[i].begin(), dg::EXP<value_type>());
}
template< class container>
void Asela< container>::log( const std::vector<container>& y, std::vector<container>& target, unsigned howmany)
{
    for( unsigned i=0; i<howmany; i++)
        thrust::transform( y[i].begin(), y[i].end(), target[i].begin(), dg::LN<value_type>());
}


}//namespace eule

