#pragma once

#include "dg/xspacelib.cuh"
#include "dg/cg.cuh"
#include "dg/gamma.cuh"

#include "parameters.h"
#include "geometry.h"

#ifdef DG_BENCHMARK
#include "dg/timer.cuh"
#endif //DG_BENCHMARK

namespace eule
{
//diffusive terms (add mu_hat?)
template<class container>
struct Rolkar
{
    Rolkar( const dg::Grid3d<double>& g, double nu_x, double nu_z):nu_perp_(nu_x), nu_parallel_(nu_z), w3d( 3, dg::create::w3d(g)), v3d( 3, dg::create::v3d(g)), temp( g.size()){
        LaplacianM_perp = dg::create::laplacianM_perp( g, dg::normed, dg::XSPACE);
        LaplacianM_para = dg::create::laplacianM_parallel( g, dg::PER);
    }
    void operator()( const std::vector<container>& x, std::vector<container>& y)
    {
        for( unsigned i=0; i<x.size(); i++)
        {
            dg::blas2::gemv( LaplacianM_perp, x[i], temp);
            dg::blas2::gemv( LaplacianM_perp, temp, y[i]);
            dg::blas1::axpby( -nu_perp_, y[i], 0., y[i]);
            dg::blas2::gemv( LaplacianM_para, x[i], temp);
            dg::blas1::axpby( -nu_parallel_, temp, 1., y[i]);
        }
        //divide( y[0], x[0], y[0]);
        //divide( y[1], x[1], y[1]);
        //divide( y[2], x[0], y[2]);
    }
    const dg::DMatrix& laplacianM()const {return LaplacianM_perp;}
    const std::vector<container>& weights(){return w3d;}
    const std::vector<container>& precond(){return v3d;}

  private:
    void divide( const container& zaehler, const container& nenner, container& result)
    {
        thrust::transform( zaehler.begin(), zaehler.end(), nenner.begin(), result.begin(), 
                thrust::divides< typename container::value_type>());
    }
    double nu_perp_, nu_parallel_;
    const std::vector<container> w3d, v3d;
    container temp;
    dg::DMatrix LaplacianM_perp;
    dg::DMatrix LaplacianM_para;
};

struct Fail : public std::exception
{

    Fail( double eps): eps( eps) {}
    double epsilon() const { return eps;}
    char const* what() const throw(){ return "Failed to converge";}
  private:
    double eps;
};

template< class container=thrust::device_vector<double> >
struct Feltor
{
    typedef std::vector<container> Vector;
    typedef typename container::value_type value_type;
    typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    //typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    typedef dg::DMatrix Matrix; //fastest device Matrix (does this conflict with 

    Feltor( const dg::Grid3d<value_type>& g, Parameters p);

    void exp( const std::vector<container>& src, std::vector<container>& dst, unsigned);

    void log( const std::vector<container>& src, std::vector<container>& dst, unsigned);

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
    //const Matrix& laplacianM( ) const { return laplaceM;}

    /**
     * @brief Return the Gamma operator used by this object
     *
     * @return Gamma operator
     */

    void operator()( const std::vector<container>& y, std::vector<container>& yp);

    double mass( ) {return mass_;}
    double mass_diffusion( ) {return diff_;}
    double energy( ) {return energy_;}
    double energy_diffusion( ){ return ediff_;}

  private:
    //use chi and omega as helpers to compute square velocity in omega
    const container& compute_vesqr( const container& potential);
    //extrapolates and solves for phi[1], then adds square velocity ( omega)
    const container& compute_psi( const container& potential);
    const container& polarisation( const std::vector<container>& y);

    container chi, omega;
    const container iris, pupil;

    std::vector<container> phi, phi_old;
    std::vector<container> expy, dzy;

    //matrices and solvers
    Matrix A, dz; 
    dg::ArakawaX< container> arakawa; 
    dg::Polarisation2dX< thrust::host_vector<value_type> > pol; //note the host vector
    dg::CG<container > pcg;

    const container w3d, v3d, one;
    const Parameters p;
    const double eps_hat;

    double mass_, energy_, diff_, ediff_;

};

template< class container>
Feltor< container>::Feltor( const dg::Grid3d<value_type>& grid, Parameters p ): 
    chi( grid.size(), 0.), omega(chi),
    iris( dg::evaluate( Iris( p.a, p.thickness), grid)), pupil( dg::evaluate( Pupil( p.a, p.thickness), grid)),
    phi( 2, chi), phi_old( phi), expy( phi), 
    dzy( 3, chi),
    dz( dg::create::dz(grid)),
    arakawa( grid), 
    pol(     grid), 
    pcg( omega, omega.size()), 
    w3d( dg::create::w3d(grid)), v3d( dg::create::v3d(grid)), one( grid.size(), 1.),
    p(p), eps_hat( 4.*M_PI*M_PI*p.a*p.a/p.eps_a/p.eps_a)
{
    //dg::create derivatives
    //laplaceM = dg::create::laplacianM( grid, normed, dg::XSPACE, dg::symmetric); //doesn't hurt to be symmetric but doesn't solver pb
    //if( !global)
    A = dg::create::laplacianM_perp( grid, dg::not_normed, dg::XSPACE, dg::symmetric);

}

template< class container>
const container& Feltor<container>::compute_vesqr( const container& potential)
{
    dg::blas2::gemv( arakawa.dx(), potential, chi);
    dg::blas2::gemv( arakawa.dy(), potential, omega);
    //dg::blas1::pointwiseDot( binv, chi, chi);
    //dg::blas1::pointwiseDot( binv, omega, omega);
    dg::blas1::pointwiseDot( chi, chi, chi);
    dg::blas1::pointwiseDot( omega, omega, omega);
    dg::blas1::axpby( 1., chi, 1.,  omega);
    return omega;
}
template< class container>
const container& Feltor<container>::compute_psi( const container& potential)
{
//    dg::blas1::axpby( 2., phi[1], -1.,  phi_old[1]);
//    phi[1].swap( phi_old[1]);
//
//    dg::blas2::symv( w3d, potential, omega);
//#ifdef DG_BENCHMARK
//    dg::Timer t;
//    t.tic();
//#endif //DG_BENCHMARK
//    unsigned number = pcg( gamma1, phi[1], omega, v3d, eps_gamma);
//    if( number == pcg.get_max())
//        throw Fail( eps_gamma);
//#ifdef DG_BENCHMARK
//    std::cout << "# of pcg iterations for psi \t"<< number << "\t";
//    t.toc();
//    std::cout<< "took \t"<<t.diff()<<"s\n";
//#endif //DG_BENCHMARK
//    //now add -0.5v_E^2
    //dg::blas1::axpby( 1., phi[1], -0.5, compute_vesqr( potential), phi[1]);
    dg::blas1::axpby( 1., potential, -0.5, compute_vesqr( potential), phi[1]);
    return phi[1];
}


//computes and modifies expy!!
template<class container>
const container& Feltor< container>::polarisation( const std::vector<container>& y)
{
    //extrapolate phi and gamma_n
    dg::blas1::axpby( 2., phi[0], -1.,  phi_old[0]);
    //dg::blas1::axpby( 2., gamma_n, -1., gamma_old);
    //dg::blas1::axpby( 1., phi[1], 0.,  phi_old[1]);
    //dg::blas1::axpby( 0., gamma_n, 0., gamma_old);
    //gamma_n.swap( gamma_old);
    phi[0].swap( phi_old[0]);

#ifdef DG_BENCHMARK
    dg::Timer t; 
    t.tic();
#endif
    //compute chi and polarisation
    exp( y, expy, 2);
    dg::blas1::axpby( 1., expy[1], 0., chi); //\chi = a_i \mu_i n_i
    //dg::blas1::pointwiseDot( binv, expy[1], chi); //\chi = n_i
    //dg::blas1::pointwiseDot( binv, chi, chi); //\chi *= binv^2
    //cusp::csr_matrix<int, double, MemorySpace> B = pol.create(chi); //first transfer to device
    //cusp::ell_matrix<int, double, cusp::host_memory> B = pol.create(chi); //first convert on host
    //A = B;  
    A = pol.create( chi);
    //compute omega
    thrust::transform( expy[0].begin(), expy[0].end(), expy[0].begin(), dg::PLUS<double>(-1)); //n_e -1
    thrust::transform( expy[1].begin(), expy[1].end(), omega.begin(), dg::PLUS<double>(-1)); //n_i -1
    //dg::blas2::symv( w3d, omega, omega); 
#ifdef DG_BENCHMARK
    t.toc();
    std::cout<< "Polarisation assembly took "<<t.diff()<<"s\n";
    t.tic();
#endif 
//    //Attention!! gamma1 wants Dirichlet BC
//    unsigned number = pcg( gamma1, gamma_n, omega, v3d, eps_gamma);
//    if( number == pcg.get_max())
//        throw Fail( eps_gamma);
//#ifdef DG_BENCHMARK
//    std::cout << "# of pcg iterations for n_i \t"<< number <<"\t";
//    t.toc();
//    std::cout<< "took \t"<<t.diff()<<"s\n";
//    t.tic();
//#endif 
//    dg::blas1::axpby( -1., expy[0], 1., gamma_n, omega); //omega = a_i\Gamma n_i - n_e
    dg::blas1::axpby( -1., expy[0], 1., omega);
    dg::blas2::symv( w3d, omega, omega);
    unsigned number = pcg( A, phi[0], omega, v3d, p.eps_pol);
    if( number == pcg.get_max())
        throw Fail( p.eps_pol);
#ifdef DG_BENCHMARK
    std::cout << "# of pcg iterations for phi \t"<< number <<"\t";
    t.toc();
    std::cout<< "took \t"<<t.diff()<<"s\n";
#endif //DG_DEBUG

    return phi[0];
}

template< class container>
void Feltor< container>::operator()( const std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 3);
    assert( y.size() == yp.size());

    phi[0] = polarisation( y);
    phi[1] = compute_psi( phi[0]);

    //update energetics, 2% of total time
    exp( y, expy, 2);
    mass_ = dg::blas2::dot( one, w3d, expy[0] ); //take real ion density which is electron density!!
    double Ue = dg::blas2::dot( y[0], w3d, expy[0]);
    double Ui = p.tau_i*dg::blas2::dot( y[1], w3d, expy[1]);
    double Uphi = 0.5*dg::blas2::dot( expy[1], w3d, omega); 
    dg::blas1::pointwiseDot( y[2], y[2], omega);
    double Upar = 0.5*p.mu_e*dg::blas2::dot( expy[1], w3d, omega); 
    energy_ = Ue + Ui + Uphi + Upar;

    arakawa( y[0], phi[0], yp[0]);
    arakawa( y[1], phi[1], yp[1]);
    arakawa( y[2], phi[0], yp[2]);

    //compute parallel derivatives
    dg::blas2::gemv( dz, y[0], dzy[0]);
    dg::blas2::gemv( dz, y[2], dzy[2]);

    dg::blas1::axpby( -1., dzy[2], 1., yp[0]);
    dg::blas1::pointwiseDot( y[2], dzy[0], omega);
    dg::blas1::axpby( -1., omega, 1., yp[0]);

    dg::blas1::pointwiseDot( y[2], dzy[2], omega);
    dg::blas1::axpby( -1., omega, 1., yp[2]);
    dg::blas1::axpby( +1./eps_hat/p.mu_e, dzy[0], 1., yp[2]);
    dg::blas2::gemv( dz, phi[0], chi);
    dg::blas1::axpby( -1./eps_hat/p.mu_e, chi, 1., yp[2]);
    //add resistivity
    dg::blas1::axpby( -p.c_hat/eps_hat/p.mu_e, y[2], 1., yp[2]);
    //apply mask functions
    for( unsigned i=0; i<3; i++)
        dg::blas1::pointwiseDot( iris, yp[i], yp[i]);
    dg::blas1::axpby( p.lnn_inner ,pupil, 1., yp[0]);
    dg::blas1::axpby( p.lnn_inner ,pupil, 1., yp[1]);

}

template< class container>
void Feltor< container>::exp( const std::vector<container>& y, std::vector<container>& target, unsigned howmany)
{
    for( unsigned i=0; i<howmany; i++)
        thrust::transform( y[i].begin(), y[i].end(), target[i].begin(), dg::EXP<value_type>());
}
template< class container>
void Feltor< container>::log( const std::vector<container>& y, std::vector<container>& target, unsigned howmany)
{
    for( unsigned i=0; i<howmany; i++)
        thrust::transform( y[i].begin(), y[i].end(), target[i].begin(), dg::LN<value_type>());
}

template< class container>
void Feltor<container>::divide( const container& zaehler, const container& nenner, container& result)
{
    thrust::transform( zaehler.begin(), zaehler.end(), nenner.begin(), result.begin(), 
            thrust::divides< typename container::value_type>());
}


} //namespace eule
