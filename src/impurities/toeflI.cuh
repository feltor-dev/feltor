#ifndef _DG_TOEFLR_CUH
#define _DG_TOEFLR_CUH

#include "dg/backend/xspacelib.cuh"
#include "dg/algorithm.h"

#ifdef DG_BENCHMARK
#include "dg/backend/timer.cuh"
#endif


//TODO es wäre besser, wenn ToeflI auch einen Zeitschritt berechnen würde 
// dann wäre die Rückgabe der Felder (Potential vs. Masse vs. exp( y)) konsistenter
// (nur das Objekt weiß welches Feld zu welchem Zeitschritt gehört)

namespace dg
{

template< class container=thrust::device_vector<double> >
struct ToeflI
{
    typedef std::vector<container> Vector;
    typedef typename container::value_type value_type;
    typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    //typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    typedef dg::DMatrix Matrix; //fastest device Matrix (does this conflict with 

    /**
     * @brief Construct a ToeflI solver object
     *
     * @param g The grid on which to operate
     * @param kappa The curvature
     * @param nu The artificial viscosity
     * @param tau The ion temperature
     * @param eps_pol stopping criterion for polarization equation
     * @param eps_gamma stopping criterion for Gamma operator
     */
    ToeflI( const Grid2d<value_type>& g, double kappa, double nu, double tau, double a_z, double mu_z, double tau_z, double eps_pol, double eps_gamma);

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
    Helmholtz<Matrix, container, container>& gamma() {return gamma1;}

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
     */
    double mass( ) {return mass_;}
    /**
     * @brief Return the last integrated mass diffusion of operator() in a global computation
     *
     * @return int \nu \Delta (exp(y[0])-1)
     */
    double mass_diffusion( ) {return diff_;}
    /**
     * @brief Return the energy of the last field in operator() in a global computation
     *
     * @return integrated total energy in {ne, ni}
     */
    double energy( ) {return energy_;}
    /**
     * @brief Return the integrated energy diffusion of the last field in operator() in a global computation
     *
     * @return integrated total energy diffusion
     */
    double energy_diffusion( ){ return ediff_;}

  private:
    //use chi and omega as helpers to compute square velocity in omega
    const container& compute_vesqr( container& potential);
    //extrapolates and solves for phi[1], then adds square velocity ( omega)
    const container& compute_psi( container& potential, int idx);
    const container& polarization( const std::vector<container>& y);

    container chi, omega;
    const container binv; //magnetic field

    std::vector<container> phi, phi_old, dyphi;
    std::vector<container> expy, dxy, dyy, lapy;
    std::vector<container> gamma_n, gamma_old;

    //matrices and solvers
    Matrix laplaceM; //contains normalized laplacian
    Helmholtz< Matrix, container, container > gamma1;
    ArakawaX< Matrix, container> arakawa; 
    dg::Elliptic< Matrix, container, container > pol; 
    CG<container > pcg;

    const container w2d, v2d, one;
    const double eps_pol, eps_gamma; 
    const double kappa, nu;
    double a_[3], mu_[3], tau_[3];

    double mass_, energy_, diff_, ediff_;

};

template< class container>
ToeflI< container>::ToeflI( const Grid2d<value_type>& grid, double kappa, double nu, double tau_i, double a_z, double mu_z, double tau_z,  double eps_pol, double eps_gamma ): 
    chi( grid.size(), 0.), omega(chi),  
    binv( evaluate( LinearX( kappa, 1.), grid)), 
    phi( 3, chi), phi_old( phi), dyphi( phi),
    gamma_n( 2, chi), gamma_old( gamma_n),
    expy( phi), dxy( expy), dyy( dxy), lapy( dyy),
    gamma1(  grid, -0.5*tau_i),
    arakawa( grid), 
    pol(     grid), 
    pcg( omega, omega.size()), 
    w2d( create::weights(grid)), v2d( create::inv_weights(grid)), one( dg::evaluate(dg::one, grid)),
    eps_pol(eps_pol), eps_gamma( eps_gamma), kappa(kappa), nu(nu)
{
    tau_[0] = -1; 
    tau_[1] = tau_i;
    tau_[2] = tau_z;
    double a_i = 1-a_z, mu_i = 1.;
    a_[0] = 1., a_[1] = a_i, a_[2] = a_z;
    mu_[0] = 0., mu_[1] = mu_i, mu_[2] = mu_z;
    //std::cout << a_[0]<<" "<<a_[1]<<" "<<a_[2]<<"\n";
    //std::cout << mu_[0]<<" "<<mu_[1]<<" "<<mu_[2]<<"\n";
    //std::cout << tau_[0]<<" "<<tau_[1]<<" "<<tau_[2]<<"\n";
    //std::cin >> tau_z;
    //create derivatives
    laplaceM = create::laplacianM( grid, normed, dg::symmetric); //doesn't hurt to be symmetric but doesn't solve pb

}

template< class container>
const container& ToeflI<container>::compute_vesqr( container& potential)
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

//idx is impurity species one or two
template< class container>
const container& ToeflI<container>::compute_psi( container& potential, int idx)
{
    //compute Gamma phi[0]
    blas1::axpby( 2., phi[idx], -1.,  phi_old[idx]);
    phi[ idx].swap( phi_old[idx]);

    blas2::symv( w2d, potential, omega);
#ifdef DG_BENCHMARK
    Timer t;
    t.tic();
#endif //DG_BENCHMARK
    gamma1.alpha() = -0.5*tau_[idx]*mu_[idx];
    unsigned number = pcg( gamma1, phi[idx], omega, v2d, eps_gamma);
    if( number == pcg.get_max())
        throw Fail( eps_gamma);
#ifdef DG_BENCHMARK
    std::cout << "# of pcg iterations for psi "<<idx<<" \t"<< number << "\t";
    t.toc();
    std::cout<< "took \t"<<t.diff()<<"s\n";
#endif //DG_BENCHMARK
    //now add -0.5v_E^2
    blas1::axpby( -0.5*mu_[idx], compute_vesqr( potential), 1., phi[idx]);
    return phi[idx];
}


//computes expy!!
template<class container>
const container& ToeflI< container>::polarization( const std::vector<container>& y)
{
    //extrapolate phi and gamma_n
    blas1::axpby( 2., phi[0], -1.,  phi_old[0]);
    phi[0].swap( phi_old[0]);
    blas1::axpby( 2., gamma_n[0], -1., gamma_old[0]);
    gamma_n[0].swap( gamma_old[0]);
    blas1::axpby( 2., gamma_n[1], -1., gamma_old[1]);
    gamma_n[1].swap( gamma_old[1]);

#ifdef DG_BENCHMARK
    Timer t; 
    t.tic();
#endif
    //compute polarizability and polarization matrix
    exp( y, expy);
    blas1::axpby( a_[1]*mu_[1], expy[1], 0., chi); //\chi = a_i \mu_i n_i + a_s \mu_s n_s
    blas1::axpby( a_[2]*mu_[2], expy[2], 1., chi);
    blas1::pointwiseDot( binv, chi, chi); 
    blas1::pointwiseDot( binv, chi, chi); //\chi *= binv^2
    pol.set_chi( chi);
#ifdef DG_BENCHMARK
    t.toc();
    std::cout<< "Polarisation assembly took "<<t.diff()<<"s\n";
    t.tic();
#endif 
    //compute \Gamma n_i
    for( unsigned i = 1; i<=2; i++)
    {
        thrust::transform( expy[i].begin(), expy[i].end(), omega.begin(), dg::PLUS<double>(-1)); //n_s -1
        blas2::symv( w2d, omega, omega); 
        //Attention!! gamma1 wants Dirichlet BC
        gamma1.alpha() = -0.5*tau_[i]*mu_[i];
        unsigned number = pcg( gamma1, gamma_n[i-1], omega, v2d, eps_gamma);
        if( number == pcg.get_max())
            throw Fail( eps_gamma);
#ifdef DG_BENCHMARK
        std::cout << "# of pcg iterations for gamma_n"<<i<<" \t"<< number <<"\t";
        t.toc();
        std::cout<< "took \t"<<t.diff()<<"s\n";
        t.tic();
#endif 
    }
    //compute charge density
    thrust::transform( expy[0].begin(), expy[0].end(), omega.begin(), dg::PLUS<double>(-1)); //n_e -1
    blas1::axpby( -1., omega, a_[1], gamma_n[0], omega); //omega = a_i\Gamma n_i - n_e
    blas1::axpby( a_[2], gamma_n[1], 1, omega); //omega += a_z \Gamma n_z
    blas2::symv( w2d, omega, omega);
    unsigned number = pcg( pol, phi[0], omega, v2d, eps_pol);
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
void ToeflI< container>::operator()(std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 3);
    assert( y.size() == yp.size());

    phi[0] = polarization( y);
    phi[1] = compute_psi( phi[0], 1);
    phi[2] = compute_psi( phi[0], 2);

    //update energetics, 2% of total time
        exp( y, expy);
        mass_ = blas2::dot( one, w2d, expy[0] ); //take real ion density which is electron density!!
        double Ue = blas2::dot( y[0], w2d, expy[0]);
        double Ui = a_[1]*tau_[1]*blas2::dot( y[1], w2d, expy[1]);
        double Uz = a_[2]*tau_[2]*blas2::dot( y[2], w2d, expy[2]);
        double Uphii = 0.5*a_[1]*mu_[1]*blas2::dot( expy[1], w2d, omega); 
        double Uphiz = 0.5*a_[2]*mu_[2]*blas2::dot( expy[2], w2d, omega); 
        energy_ = Ue + Ui + Uphii + Uphiz;

        for( unsigned i=0; i<y.size(); i++)
        {
            thrust::transform( expy[i].begin(), expy[i].end(), expy[i].begin(), dg::PLUS<double>(-1));
            blas2::gemv( laplaceM, expy[i], lapy[i]); //Laplace wants Dir BC!!
        }
        diff_ = -nu*blas2::dot( one, w2d, lapy[0]);
        double Gi[3];
        Gi[0] = - blas2::dot( one, w2d, lapy[0]) - blas2::dot( lapy[0], w2d, y[0]); // minus 
        for( unsigned i=1; i<3; i++)
            Gi[i] = - tau_[i]*(blas2::dot( one, w2d, lapy[i]) + blas2::dot( lapy[i], w2d, y[i])); // minus 
        double Gphi[3];
        for( unsigned i=0; i<3; i++)
            Gphi[i] = -blas2::dot( phi[i], w2d, lapy[i]);
        //std::cout << "ge "<<Ge<<" gi "<<Gi<<" gphi "<<Gphi<<" gpsi "<<Gpsi<<"\n";
        ediff_ = nu*( Gi[0] - Gphi[0] + a_[1]*(Gi[1] + Gphi[1]) + a_[2]*( Gi[2] + Gphi[2]));

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
        blas1::axpby( kappa, dyphi[i], 1., yp[i]);
        blas1::axpby( tau_[i]*kappa, dyy[i], 1., yp[i]);
    }

    //add laplacians
    for( unsigned i=0; i<y.size(); i++)
    {
        blas2::gemv( laplaceM, y[i], lapy[i]);
            blas1::pointwiseDot( dxy[i], dxy[i], dxy[i]);
            blas1::pointwiseDot( dyy[i], dyy[i], dyy[i]);
            //now sum all 3 terms up 
            blas1::axpby( -1., dyy[i], 1., lapy[i]); //behold the minus
            blas1::axpby( -1., dxy[i], 1., lapy[i]); //behold the minus
        blas1::axpby( -nu, lapy[i], 1., yp[i]); //rescale 
    }

}

template< class container>
void ToeflI< container>::exp( const std::vector<container>& y, std::vector<container>& target)
{
    for( unsigned i=0; i<y.size(); i++)
        thrust::transform( y[i].begin(), y[i].end(), target[i].begin(), dg::EXP<value_type>());
}
template< class container>
void ToeflI< container>::log( const std::vector<container>& y, std::vector<container>& target)
{
    for( unsigned i=0; i<y.size(); i++)
        thrust::transform( y[i].begin(), y[i].end(), target[i].begin(), dg::LN<value_type>());
}

template< class container>
void ToeflI<container>::divide( const container& zaehler, const container& nenner, container& result)
{
    thrust::transform( zaehler.begin(), zaehler.end(), nenner.begin(), result.begin(), 
            thrust::divides< typename container::value_type>());
}

}//namespace dg

#endif //_DG_TOEFLR_CUH
