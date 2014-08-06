#ifndef _DG_TOEFLR_CUH
#define _DG_TOEFLR_CUH

#include <exception>

#include "dg/algorithm.h"
#include "dg/backend/polarisation.cuh"
#include "dg/backend/typedefs.cuh"

#ifdef DG_BENCHMARK
#include "dg/backend/timer.cuh"
#endif


//TODO es wäre besser, wenn ToeflR auch einen Zeitschritt berechnen würde 
// dann wäre die Rückgabe der Felder (Potential vs. Masse vs. exp( y)) konsistenter
// (nur das Objekt weiß welches Feld zu welchem Zeitschritt gehört)

namespace dg
{
template<class container>
struct Diffusion
{
    Diffusion( const dg::Grid2d<double>& g, double nu, bool global):nu_(nu), global(global), w2d( dg::create::w2d(g)), v2d( dg::create::v2d(g)), temp( g.size()), expx(temp){
        LaplacianM_perp = dg::create::laplacianM( g, dg::normed);
    }
    void operator()( const std::vector<container>& x, std::vector<container>& y)
    {
        for( unsigned i=0; i<x.size(); i++)
        {
            //if( global)
            //{
            //    thrust::transform( x[i].begin(), x[i].end(), expx.begin(), dg::EXP<typename container::value_type>());
            //    dg::blas2::gemv( LaplacianM_perp, expx, temp);
            //    //dg::blas2::gemv( LaplacianM_perp, temp, y[i]);
            //    dg::blas1::axpby( -nu_, temp, 0., y[i]);
            //    divide( y[i], expx, y[i]);
            //}
            //else
            {
                dg::blas2::gemv( LaplacianM_perp, x[i], temp);
                dg::blas2::gemv( LaplacianM_perp, temp, y[i]);
                dg::blas1::axpby( -nu_, y[i], 0., y[i]);
            }
        }
    }
    const dg::DMatrix& laplacianM()const {return LaplacianM_perp;}
    const container& weights(){return w2d;}
    const container& precond(){return v2d;}

  private:
    void divide( const container& zaehler, const container& nenner, container& result)
    {
        thrust::transform( zaehler.begin(), zaehler.end(), nenner.begin(), result.begin(), 
                thrust::divides< typename container::value_type>());
    }
    double nu_;
    bool global;
    const container w2d, v2d;
    container temp, expx;
    dg::DMatrix LaplacianM_perp;
};

template< class container=thrust::device_vector<double> >
struct ToeflR
{
    typedef std::vector<container> Vector;
    typedef typename container::value_type value_type;
    typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    //typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    typedef dg::DMatrix Matrix; //fastest device Matrix (does this conflict with 
    //typedef in ArakawaX ??

    /**
     * @brief Construct a ToeflR solver object
     *
     * @param g The grid on which to operate
     * @param kappa The curvature
     * @param nu The artificial viscosity
     * @param tau The ion temperature
     * @param eps_pol stopping criterion for polarisation equation
     * @param eps_gamma stopping criterion for Gamma operator
     * @param global local or global computation
     */
    ToeflR( const Grid2d<value_type>& g, double kappa, double nu, double tau, double eps_pol, double eps_gamma, int global);

    /**
     * @brief Exponentiate pointwise every Vector in src 
     *
     * @param src source
     * @param dst destination may equal source
     */
    void exp( const std::vector<container>& src, std::vector<container>& dst);

    /**
     * @brief Take the natural logarithm pointwise of every Vector in src 
     *
     * @param src source
     * @param dst destination may equal source
     */
    void log( const std::vector<container>& src, std::vector<container>& dst);

    void divide( const container& zaehler, const container& nenner, container& result);

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
    dg::Helmholtz<Matrix, container, container >&  gamma() {return gamma1;}

    /**
     * @brief Compute the right-hand side of the toefl equations
     *
     * @param y input vector
     * @param yp the rhs yp = f(y)
     */
    void operator()( std::vector<container>& y, std::vector<container>& yp);

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
    const container binv; //magnetic field

    std::vector<container> phi, phi_old, dyphi;
    std::vector<container> expy, dxy, dyy, lapy;

    //matrices and solvers
    Matrix A; //contains unnormalized laplacian if local
    Matrix laplaceM; //contains normalized laplacian
    Helmholtz< Matrix, container, container > gamma1;
    ArakawaX< Matrix, container> arakawa; 
    Polarisation2dX< thrust::host_vector<value_type>, dg::HMatrix > pol; //note the host vector
    CG<container > pcg;

    const container w2d, v2d, one;
    const double eps_pol, eps_gamma; 
    const double kappa, nu, tau;
    const int global;

    double mass_, energy_, diff_, ediff_;

};

template< class container>
ToeflR< container>::ToeflR( const Grid2d<value_type>& grid, double kappa, double nu, double tau, double eps_pol, double eps_gamma, int global ): 
    chi( grid.size(), 0.), omega(chi), gamma_n( chi), gamma_old( chi), 
    binv( evaluate( LinearX( kappa, 1.), grid)), 
    phi( 2, chi), phi_old( phi), dyphi( phi),
    expy( phi), dxy( expy), dyy( dxy), lapy( dyy),
    gamma1(  laplaceM, w2d,v2d, -0.5*tau),
    arakawa( grid), 
    pol(     grid), 
    pcg( omega, omega.size()), 
    w2d( create::w2d(grid)), v2d( create::v2d(grid)), one( grid.size(), 1.),
    eps_pol(eps_pol), eps_gamma( eps_gamma), kappa(kappa), nu(nu), tau( tau), global( global)
{
    //create derivatives
    laplaceM = create::laplacianM( grid, normed, dg::symmetric); //doesn't hurt to be symmetric but doesn't solver pb
    //if( !global)
    A = create::laplacianM( grid, not_normed, dg::symmetric);

}

template< class container>
const container& ToeflR<container>::compute_vesqr( const container& potential)
{
    assert( global);
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
const container& ToeflR<container>::compute_psi( const container& potential)
{
    //compute Gamma phi[0]
    blas1::axpby( 2., phi[1], -1.,  phi_old[1]);
    phi[1].swap( phi_old[1]);

    blas2::symv( w2d, potential, omega);
#ifdef DG_BENCHMARK
    Timer t;
    t.tic();
#endif //DG_BENCHMARK
    unsigned number = pcg( gamma1, phi[1], omega, v2d, eps_gamma);
    if( number == pcg.get_max())
        throw Fail( eps_gamma);
#ifdef DG_BENCHMARK
    std::cout << "# of pcg iterations for psi \t"<< number << "\t";
    t.toc();
    std::cout<< "took \t"<<t.diff()<<"s\n";
#endif //DG_BENCHMARK
    //now add -0.5v_E^2
    if( global)
    {
        blas1::axpby( 1., phi[1], -0.5, compute_vesqr( potential), phi[1]);
    }
    return phi[1];
}


//computes and modifies expy!!
template<class container>
const container& ToeflR< container>::polarisation( const std::vector<container>& y)
{
    //extrapolate phi and gamma_n
    blas1::axpby( 2., phi[0], -1.,  phi_old[0]);
    blas1::axpby( 2., gamma_n, -1., gamma_old);
    //blas1::axpby( 1., phi[1], 0.,  phi_old[1]);
    //blas1::axpby( 0., gamma_n, 0., gamma_old);
    gamma_n.swap( gamma_old);
    phi[0].swap( phi_old[0]);

#ifdef DG_BENCHMARK
    Timer t; 
    t.tic();
#endif
    //compute chi and polarisation
    if( global) 
    {
        exp( y, expy);
        //blas1::axpby( 1., expy[1], 0., chi); //\chi = a_i \mu_i n_i
        blas1::pointwiseDot( binv, expy[1], chi); //\chi = n_i
        blas1::pointwiseDot( binv, chi, chi); //\chi *= binv^2
        //cusp::csr_matrix<int, double, MemorySpace> B = pol.create(chi); //first transfer to device
        //cusp::ell_matrix<int, double, cusp::host_memory> B = pol.create(chi); //first convert on host
        //A = B;  
        if( global == 1)
            A = pol.create( chi);
        //compute omega
        thrust::transform( expy[0].begin(), expy[0].end(), expy[0].begin(), dg::PLUS<double>(-1)); //n_e -1
        thrust::transform( expy[1].begin(), expy[1].end(), omega.begin(), dg::PLUS<double>(-1)); //n_i -1
    }
    else
    {
        blas1::axpby( 1., y[1], 0., omega); //n_i = omega
    }
    blas2::symv( w2d, omega, omega); 
#ifdef DG_BENCHMARK
    t.toc();
    std::cout<< "Polarisation assembly took "<<t.diff()<<"s\n";
    t.tic();
#endif 
    //Attention!! gamma1 wants Dirichlet BC
    unsigned number = pcg( gamma1, gamma_n, omega, v2d, eps_gamma);
    if( number == pcg.get_max())
        throw Fail( eps_gamma);
#ifdef DG_BENCHMARK
    std::cout << "# of pcg iterations for n_i \t"<< number <<"\t";
    t.toc();
    std::cout<< "took \t"<<t.diff()<<"s\n";
    t.tic();
#endif 
    if( global)
    {
        blas1::axpby( -1., expy[0], 1., gamma_n, omega); //omega = a_i\Gamma n_i - n_e
        if( global == 2)
            divide( omega, expy[1], omega);
        blas2::symv( w2d, omega, omega);
    }
    else
    {
        blas1::axpby( -1, y[0], 1., gamma_n, chi); 
        gamma1.alpha() = -tau;
        blas2::symv( gamma1, chi, omega); //apply \Gamma_0^-1 ( gamma_n - n_e)
        gamma1.alpha() = -0.5*tau;
    }
    number = pcg( A, phi[0], omega, v2d, eps_pol);
    if( number == pcg.get_max())
        throw Fail( eps_pol);
#ifdef DG_BENCHMARK
    std::cout << "# of pcg iterations for phi \t"<< number <<"\t";
    t.toc();
    std::cout<< "took \t"<<t.diff()<<"s\n";
#endif //DG_DEBUG

    return phi[0];
}

template< class container>
void ToeflR< container>::operator()( std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 2);
    assert( y.size() == yp.size());

    phi[0] = polarisation( y);
    phi[1] = compute_psi( phi[0]);

    //update energetics, 2% of total time
    if( global)
    {
        exp( y, expy);
        mass_ = blas2::dot( one, w2d, expy[0] ); //take real ion density which is electron density!!
        double Ue = blas2::dot( y[0], w2d, expy[0]);
        double Ui = tau*blas2::dot( y[1], w2d, expy[1]);
        double Uphi = 0.5*blas2::dot( expy[1], w2d, omega); 
        energy_ = Ue + Ui + Uphi;

        for( unsigned i=0; i<y.size(); i++)
        {
            thrust::transform( expy[i].begin(), expy[i].end(), expy[i].begin(), dg::PLUS<double>(-1));
            blas2::gemv( laplaceM, expy[i], lapy[i]); //Laplace wants Dir BC!!
        }
        diff_ = -nu*blas2::dot( one, w2d, lapy[0]);
        double Ge = - blas2::dot( one, w2d, lapy[0]) - blas2::dot( lapy[0], w2d, y[0]); // minus 
        double Gi = - tau*(blas2::dot( one, w2d, lapy[1]) + blas2::dot( lapy[1], w2d, y[1])); // minus 
        double Gphi = -blas2::dot( phi[0], w2d, lapy[0]);
        double Gpsi = -blas2::dot( phi[1], w2d, lapy[1]);
        //std::cout << "ge "<<Ge<<" gi "<<Gi<<" gphi "<<Gphi<<" gpsi "<<Gpsi<<"\n";
        ediff_ = nu*( Ge + Gi - Gphi + Gpsi);
    }

    for( unsigned i=0; i<y.size(); i++)
    {
        arakawa( y[i], phi[i], yp[i]);
        blas1::pointwiseDot( binv, yp[i], yp[i]);
    }

    //compute derivatives
    for( unsigned i=0; i<y.size(); i++)
    {
        blas2::gemv( arakawa.dx(), y[i], dxy[i]);
        blas2::gemv( arakawa.dy(), y[i], dyy[i]);
        blas2::gemv( arakawa.dy(), phi[i], dyphi[i]);
    }
    // curvature terms
    blas1::axpby( kappa, dyphi[0], 1., yp[0]);
    blas1::axpby( kappa, dyphi[1], 1., yp[1]);
    blas1::axpby( -1.*kappa, dyy[0], 1., yp[0]);
    blas1::axpby( tau*kappa, dyy[1], 1., yp[1]);

    //add laplacians
    for( unsigned i=0; i<y.size(); i++)
    {
        blas2::gemv( laplaceM, y[i], lapy[i]);
        if( global)
        {
            blas1::pointwiseDot( dxy[i], dxy[i], dxy[i]);
            blas1::pointwiseDot( dyy[i], dyy[i], dyy[i]);
            //now sum all 3 terms up 
            blas1::axpby( -1., dyy[i], 1., lapy[i]); //behold the minus
            blas1::axpby( -1., dxy[i], 1., lapy[i]); //behold the minus
        }
        //blas1::axpby( -nu, lapy[i], 1., yp[i]); //rescale 
    }

}

template< class container>
void ToeflR< container>::exp( const std::vector<container>& y, std::vector<container>& target)
{
    for( unsigned i=0; i<y.size(); i++)
        dg::blas1::transform(y[i], target[i], dg::EXP<value_type>());
        //thrust::transform( y[i].begin(), y[i].end(), target[i].begin(), dg::EXP<value_type>());
}
template< class container>
void ToeflR< container>::log( const std::vector<container>& y, std::vector<container>& target)
{
    for( unsigned i=0; i<y.size(); i++)
        dg::blas1::transform( y[i], target[i], dg::LN<value_type>());
        //thrust::transform( y[i].begin(), y[i].end(), target[i].begin(), dg::LN<value_type>());
}

template< class container>
void ToeflR<container>::divide( const container& zaehler, const container& nenner, container& result)
{
    dg::blas1::pointwiseDivide( zaehler, nenner, result);

    //thrust::transform( zaehler.begin(), zaehler.end(), nenner.begin(), result.begin(), 
            //thrust::divides< typename container::value_type>());
}

}//namespace dg

#endif //_DG_TOEFLR_CUH
