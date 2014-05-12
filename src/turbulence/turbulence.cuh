#ifndef _DG_TOEFLR_CUH
#define _DG_TOEFLR_CUH

#include <exception>

#include "dg/xspacelib.cuh"
#include "dg/average.cuh"
#include "dg/cg.cuh"
#include "dg/gamma.cuh"

#ifdef DG_BENCHMARK
#include "dg/timer.cuh"
#endif


//TODO es wäre besser, wenn Turbulence auch einen Zeitschritt berechnen würde 
// dann wäre die Rückgabe der Felder (Potential vs. Masse vs. exp( y)) konsistenter
// (nur das Objekt weiß welches Feld zu welchem Zeitschritt gehört)

namespace dg
{
struct Fail : public std::exception
{

    Fail( double eps): eps( eps) {}
    double epsilon() const { return eps;}
    char const* what() const throw(){ return "Failed to converge";}
  private:
    double eps;
};

template< class container=thrust::device_vector<double> >
struct Turbulence
{
    typedef std::vector<container> Vector;
    typedef typename container::value_type value_type;
    typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    //typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    typedef dg::DMatrix Matrix; //fastest device Matrix (does this conflict with 
    //typedef in ArakawaX ??

    /**
     * @brief Construct a Turbulence solver object
     *
     * @param g The grid on which to operate
     * @param kappa The curvature
     * @param nu The artificial viscosity
     * @param tau The ion temperature
     * @param eps_pol stopping criterion for polarisation equation
     * @param eps_gamma stopping criterion for Gamma operator
     */
    Turbulence( const Grid2d<value_type>& g, double kappa, double nu, double tau, double eps_pol, double eps_gamma, double gradient, double d);

    const container& gradient(){ return gradient_;}

    /**
     * @brief Take the natural logarithm pointwise of every Vector in src 
     *
     * @param src source
     * @param dst destination may equal source
     */
    void log( const std::vector<container>& src, std::vector<container>& dst);

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
    //const Helmholtz<Matrix, container >&  gamma() const {return gamma1;}

    /**
     * @brief Compute the right-hand side of the toefl equations
     *
     * @param y input vector (without background gradient)
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
    //use chi and omega as helpers to compute square velocity in omega
    const container& compute_vesqr( const container& potential);
    //extrapolates and solves for phi[1], then adds square velocity ( omega)
    const container& compute_psi( const container& potential);
    const container& polarisation( const std::vector<container>& y);

    container chi, omega;
    container gamma_n, gamma_old;
    const container binv, gradient_; //magnetic field
    const double grad_;

    std::vector<container> phi, phi_old, dyphi;
    std::vector<container> ypg, dyy, lapy;

    //matrices and solvers
    Matrix A; //contains polarisation matrix
    Matrix laplaceM; //contains normalized laplacian
    ArakawaX< container> arakawa; 
    Polarisation2dX< thrust::host_vector<value_type> > pol; //note the host vector
    CG<container > pcg;
    PoloidalAverage<container, thrust::device_vector<int> > average;

    const container w2d, v2d, one;
    const double eps_pol, eps_gamma; 
    const double kappa, nu, tau, d_;
    Helmholtz< Matrix, container > gamma1;

    double mass_, energy_, diff_, ediff_;

};

template< class container>
Turbulence< container>::Turbulence( const Grid2d<value_type>& grid, double kappa, double nu, double tau, double eps_pol, double eps_gamma, double gradient, double d): 
    chi( grid.size(), 0.), omega(chi), gamma_n( chi), gamma_old( chi), 
    binv( evaluate( LinearX( kappa, 1.), grid)), gradient_( evaluate( LinearX( -gradient, 1+gradient*grid.lx()), grid)), grad_(gradient),
    phi( 2, chi), phi_old( phi), dyphi( phi),
    ypg( phi), dyy( phi), lapy( dyy),
    arakawa( grid), 
    pol(     grid), 
    pcg( omega, omega.size()), 
    average( grid),
    w2d( create::w2d(grid)), v2d( create::v2d(grid)), one( grid.size(), 1.),
    eps_pol(eps_pol), eps_gamma( eps_gamma), kappa(kappa), nu(nu), tau( tau), d_(d),
    gamma1(  laplaceM, w2d, v2d, -0.5*tau)
{
    //create derivatives
    laplaceM = create::laplacianM( grid, normed, XSPACE, symmetric);
}

template< class container>
const container& Turbulence<container>::compute_vesqr( const container& potential)
{
    blas2::gemv( arakawa.dx(), potential, chi);
    blas2::gemv( arakawa.dy(), potential, omega);
    blas1::pointwiseDot( binv, chi, chi);
    blas1::pointwiseDot( binv, omega, omega);
    blas1::pointwiseDot( chi, chi, chi);
    blas1::pointwiseDot( omega, omega, omega);
    blas1::axpby( 1., chi, 1.,  omega);
    return omega;
}
template< class container>
const container& Turbulence<container>::compute_psi( const container& potential)
{
    //compute Gamma phi[0]
    blas1::axpby( 2., phi[1], -1.,  phi_old[1]);
    phi[1].swap( phi_old[1]);

    blas2::symv( w2d, potential, omega);
#ifdef DG_BENCHMARK
    Timer t;
    t.tic();
#endif //DG_BENCHMARK
    //unsigned number = pcg( gamma1, phi[1], omega, v2d, eps_gamma);
    //if( number == pcg.get_max())
        //throw Fail( eps_gamma);
#ifdef DG_BENCHMARK
    //std::cout << "# of pcg iterations for psi \t"<< number << "\t";
    t.toc();
    std::cout<< "took \t"<<t.diff()<<"s\n";
#endif //DG_BENCHMARK
    //now add -0.5v_E^2
    blas1::axpby( 1., phi[1], -0.5, compute_vesqr( potential), phi[1]);
    return phi[1];
}


template<class container>
const container& Turbulence< container>::polarisation( const std::vector<container>& y)
{
    //extrapolate phi and gamma_n
    blas1::axpby( 2., phi[0], -1.,  phi_old[0]);
    blas1::axpby( 2., gamma_n, -1., gamma_old);
    gamma_n.swap( gamma_old);
    phi[0].swap( phi_old[0]);

#ifdef DG_BENCHMARK
    Timer t; 
    t.tic();
#endif
    //compute omega
    blas2::symv( w2d, y[1], omega); 
    //Attention!! gamma1 wants Dirichlet BC
    //unsigned number = pcg( gamma1, gamma_n, omega, v2d, eps_gamma);
    //if( number == pcg.get_max())
        //throw Fail( eps_gamma);
    blas1::axpby( -1., y[0], 1., gamma_n, omega); //omega = a_i\Gamma n_i - n_e
#ifdef DG_BENCHMARK
    //std::cout << "# of pcg iterations for n_i \t"<< number <<"\t";
    t.toc();
    std::cout<< "took \t"<<t.diff()<<"s\n";
    t.tic();
#endif 
    //compute chi and polarisation
    blas1::axpby(1., y[1], 1, gradient_, chi);//chi = n_i + gradient
    blas1::pointwiseDot( binv, chi, chi); 
    blas1::pointwiseDot( binv, chi, chi); //\chi *= binv^2
    A = pol.create( chi);
#ifdef DG_BENCHMARK
    t.toc();
    std::cout<< "Polarisation assembly took "<<t.diff()<<"s\n";
    t.tic();
#endif 
    blas2::symv( w2d, omega, omega);
    unsigned number = pcg( A, phi[0], omega, v2d, eps_pol);
    if( number == pcg.get_max())
        throw Fail( eps_pol);
#ifdef DG_BENCHMARK
    std::cout << "# of pcg iterations for phi \t"<< number <<"\t";
    t.toc();
    std::cout<< "took \t"<<t.diff()<<"s\n";
#endif //DG_DEBUG

    return phi[0];
}

//y is the density without background gradient (such that the BC is zero)
//background is -gx + 1 + gl_x
template< class container>
void Turbulence< container>::operator()( const std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 2);
    assert( y.size() == yp.size());

    phi[0] = polarisation( y);
    phi[1] = compute_psi( phi[0]);

    for( unsigned i=0; i<y.size(); i++)
    {
        arakawa( y[i], phi[i], yp[i]);
        blas1::pointwiseDot( binv, yp[i], yp[i]);
    }

    //compute derivatives
    for( unsigned i=0; i<y.size(); i++)
    {
        blas2::gemv( arakawa.dy(), y[i], dyy[i]);
        blas2::gemv( arakawa.dy(), phi[i], dyphi[i]);
        blas1::axpby( -grad_, dyphi[i], 1., yp[i]); //-g\partial_y \phi
        blas1::axpby( 1, y[i], 1, gradient_, ypg[i]);
        blas1::pointwiseDot( dyphi[i], ypg[i], dyphi[i]); //dyphi <- dyphi*n_e
        blas1::axpby( kappa, dyphi[i], 1., yp[i]);
    }
    // curvature terms
    blas1::axpby( -1.*kappa, dyy[0], 1., yp[0]);
    blas1::axpby( tau*kappa, dyy[1], 1., yp[1]);
    //add HW coupling to n_e
    //blas1::pointwiseDot( phi[0], ypg[0], chi);
    //blas1::axpby( d_, chi, 1, yp[0]);
    //average( chi, omega);
    //blas1::axpby( -d_, omega, 1., yp[0]);
    blas1::axpby( d_, phi[0], 1, yp[0]);
    average( phi[0], chi);
    blas1::axpby( -d_, chi, 1., yp[0]);

    blas1::axpby( -d_, y[0], 1, yp[0]);
    average( y[0], chi);
    blas1::axpby( d_, chi, 1., yp[0]);

    //add laplacians
    for( unsigned i=0; i<y.size(); i++)
    {
        blas2::gemv( laplaceM, y[i], lapy[i]);
        blas1::axpby( -nu, lapy[i], 1., yp[i]); //rescale 
    }

}

template< class container>
void Turbulence< container>::log( const std::vector<container>& y, std::vector<container>& target)
{
    for( unsigned i=0; i<y.size(); i++)
        thrust::transform( y[i].begin(), y[i].end(), target[i].begin(), dg::LN<value_type>());
}

}//namespace dg

#endif //_DG_TOEFLR_CUH
