#ifndef _DG_TOEFLR_CUH
#define _DG_TOEFLR_CUH

#include <exception>

#include "dg/xspacelib.cuh"
#include "dg/cg.cuh"
#include "dg/gamma.cuh"

#ifdef DG_BENCHMARK
#include "dg/timer.cuh"
#endif



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
struct ToeflR
{
    typedef std::vector<container> Vector;
    typedef typename container::value_type value_type;
    typedef typename thrust::iterator_space<typename container::iterator>::type MemorySpace;
    typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;

    ToeflR( const Grid<value_type>& g, double kappa, double nu, double tau, double eps_pol, double eps_gamma, bool global);

    void exp( const std::vector<container>& y, std::vector<container>& target);

    void log( const std::vector<container>& y, std::vector<container>& target);

    const std::vector<container>& potential( ) const { return phi;}

    const Matrix& laplacianM( ) const { return laplaceM;}

    const Gamma<Matrix, container >&  gamma() const {return gamma1;}

    const container& compute_psi( const container& potential);

    const container& compute_vesqr( const container& potential);

    void operator()( const std::vector<container>& y, std::vector<container>& yp);

    double energy( const std::vector<container>& y, const container& potential);

    double energy_dot( const std::vector<container>& y, const std::vector<container>& potential);

  private:
    const container& polarisation( const std::vector<container>& y);

    container chi, omega;
    container gamma_n, gamma_old;
    const container binv;

    std::vector<container> phi, phi_old, dyphi;
    std::vector<container> expy, dxy, dyy, lapy;

    Matrix A; //contains unnormalized laplacian if local
    Matrix laplaceM; //contains normalized laplacian
    Gamma< Matrix, container > gamma1;
    ArakawaX< container> arakawa; 
    Polarisation2dX< thrust::host_vector<value_type> > pol; //note the host vector
    CG<container > pcg;

    const container w2d, v2d;
    double eps_pol, eps_gamma; 
    double kappa, nu, tau;
    bool global;

};

template< class container>
ToeflR< container>::ToeflR( const Grid<value_type>& grid, double kappa, double nu, double tau, double eps_pol, double eps_gamma, bool global ): 
    chi( grid.size(), 0.), omega(chi), gamma_n( chi), gamma_old( chi), 
    binv( evaluate( LinearX( kappa, 1.), grid)), 
    phi( 2, chi), phi_old( phi), dyphi( phi),
    expy( phi), dxy( expy), dyy( dxy), lapy( dyy),
    gamma1(  laplaceM, w2d, -0.5*tau),
    arakawa( grid), 
    pol(     grid), 
    pcg( omega, omega.size()), 
    w2d( create::w2d(grid)), v2d( create::v2d(grid)),
    eps_pol(eps_pol), eps_gamma( eps_gamma), kappa(kappa), nu(nu), tau( tau), global( global)
{
    //create derivatives
    laplaceM = create::laplacianM( grid, normed);
    if( !global)
        A = create::laplacianM( grid, not_normed);

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
#endif //DG_DEBUG
    unsigned number = pcg( gamma1, phi[1], omega, v2d, eps_gamma);
    if( number == pcg.get_max())
        throw Fail( eps_gamma);
#ifdef DG_BENCHMARK
    std::cout << "Number of pcg iterations2 "<< number <<std::endl;
    t.toc();
    std::cout<< "took "<<t.diff()<<"s\n";
#endif //DG_DEBUG
    //now add -0.5v_E^2
    if( global)
    {
        blas1::axpby( 1., phi[1], -0.5, compute_vesqr( potential), phi[1]);
        //cudaDeviceSynchronize();
    }
    return phi[1];
}

template< class container>
double ToeflR< container>::energy( const std::vector<container>& y, const container& potential)
{
    assert( global);
    exp( y, expy); // y-> ln(n), expy -> n
    double Ue = blas2::dot( y[0], w2d, expy[0]);
    double Ui = tau*blas2::dot( y[1], w2d, expy[1]);
    omega = compute_vesqr( potential);
    double Uphi = 0.5*blas2::dot( expy[1], w2d, omega); 
    //std::cout << "ue "<<Ue<<" ui "<<Ui<<" uphi "<<Uphi<<"\n";
    return Ue + Ui + Uphi;
}

template< class container>
double ToeflR< container>::energy_dot( const std::vector<container>& y, const std::vector<container>& potential)
{
    assert( global);
    container one( y[0].size(), 1.);
    exp( y, expy); // y-> ln(n), expy -> n
    for( unsigned i=0; i<y.size(); i++)
    {
        thrust::transform( expy[i].begin(), expy[i].end(), expy[i].begin(), dg::PLUS<double>(-1));
        blas2::gemv( laplaceM, expy[i], lapy[i]); //DOESNT WORK
        //thrust::transform( lapy[i].begin(), lapy[i].end(), lapy[i].begin(), dg::PLUS<double>(+1));
    }
    double Ge = - blas2::dot( one, w2d, lapy[0]) - blas2::dot( lapy[0], w2d, y[0]); // minus 
    double Gi = - tau*(blas2::dot( one, w2d, lapy[1]) + blas2::dot( lapy[1], w2d, y[1])); // minus 
    double Gphi = -blas2::dot( potential[0], w2d, lapy[0]);
    double Gpsi = -blas2::dot( potential[1], w2d, lapy[1]);
    //std::cout << "ge "<<Ge<<" gi "<<Gi<<" gphi "<<Gphi<<" gpsi "<<Gpsi<<"\n";
    return nu*( Ge + Gi - Gphi + Gpsi);
}

//how to set up a computation?
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
        cusp::csr_matrix<int, double, MemorySpace> B = pol.create(chi); //first transfer to device
        A = B; 
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
    std::cout << "Number of pcg iterations0 "<< number <<std::endl;
    t.toc();
    std::cout<< "took "<<t.diff()<<"s\n";
    t.tic();
#endif 
    if( global)
    {
        blas1::axpby( -1., expy[0], 1., gamma_n, omega); //omega = a_i\Gamma n_i - n_e
        blas2::symv( w2d, omega, omega);
    }
    else
    {
        blas1::axpby( -1, y[0], 1., gamma_n, chi); 
        //container dxchi(chi),dxxchi( chi),  dychi( chi), dyychi(chi);
        //blas2::gemv( arakawa.dx(), chi, dxchi);
        //blas2::gemv( arakawa.dx(), dxchi, dxxchi);
        //blas2::gemv( arakawa.dy(), chi, dychi);
        //blas2::gemv( arakawa.dy(), dychi, dyychi);
        //blas1::axpby( 1., chi, -tau, dxxchi, omega); 
        //blas1::axpby( 1., omega, -tau, dyychi, omega);
        //blas2::symv( w2d, omega, omega);

        gamma1.alpha() = -tau;
        blas2::symv( gamma1, chi, omega); //apply \Gamma_0^-1 ( gamma_n - n_e)
        gamma1.alpha() = -0.5*tau;
    }
    number = pcg( A, phi[0], omega, v2d, eps_pol);
    if( number == pcg.get_max())
        throw Fail( eps_pol);
#ifdef DG_BENCHMARK
    std::cout << "Number of pcg iterations1 "<< number <<std::endl;
    t.toc();
    std::cout<< "took "<<t.diff()<<"s\n";
    t.tic();
#endif //DG_DEBUG

    return phi[0];
}

template< class container>
void ToeflR< container>::operator()( const std::vector<container>& y, std::vector<container>& yp)
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
        blas1::axpby( -nu, lapy[i], 1., yp[i]); //rescale 
    }
    //cudaDeviceSynchronize();

}

template< class container>
void ToeflR< container>::exp( const std::vector<container>& y, std::vector<container>& target)
{
    for( unsigned i=0; i<y.size(); i++)
        thrust::transform( y[i].begin(), y[i].end(), target[i].begin(), dg::EXP<value_type>());
}
template< class container>
void ToeflR< container>::log( const std::vector<container>& y, std::vector<container>& target)
{
    for( unsigned i=0; i<y.size(); i++)
        thrust::transform( y[i].begin(), y[i].end(), target[i].begin(), dg::LN<value_type>());
}

}//namespace dg

#endif //_DG_TOEFLR_CUH
