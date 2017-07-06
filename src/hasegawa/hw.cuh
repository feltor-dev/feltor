#pragma once

#include <exception>

#include "dg/algorithm.h"
#include "dg/average.h"

#ifdef DG_BENCHMARK
#include "dg/backend/timer.cuh"
#endif

///@note This is an old copy of the toefl project and shouldn't be taken as a basis for a new project

namespace dg
{

template< class Matrix, class container>
struct Diffusion
{
    Diffusion( const dg::Grid2d& g, double nu): nu_(nu),
        w2d(dg::create::weights( g)), v2d( dg::create::inv_weights(g)), temp( g.size()), LaplacianM( g, dg::normed, dg::centered) {
        }
    void operator()( const std::vector<container>& x, std::vector<container>& y)
    {
        dg::blas1::axpby( 0., x, 0, y);
        for( unsigned i=0; i<x.size(); i++)
        {
            dg::blas2::gemv( LaplacianM, x[i], temp);
            dg::blas2::gemv( LaplacianM, temp, y[i]);
            //dg::blas2::gemv( LaplacianM, y[i], temp);
            //dg::blas1::axpby( 0., y[i], -nu_ , y[i]);
            dg::blas1::scal( y[i], -nu_);
        }
    }
    const container& weights(){return w2d;}
    const container& precond(){return v2d;}
  private:
    double nu_;
    const container w2d, v2d;
    container temp;
    Elliptic<dg::CartesianGrid2d, Matrix, container> LaplacianM;
};


template< class Matrix, class container=thrust::device_vector<double> >
struct HW
{
    typedef std::vector<container> Vector;
    typedef typename container::value_type value_type;
    typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    //typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;

    /**
     * @brief Construct a HW solver object
     *
     * @param g The grid on which to operate
     * @param kappa The curvature
     * @param nu The artificial viscosity
     * @param tau The ion temperature
     * @param eps_pol stopping criterion for polarisation equation
     * @param eps_gamma stopping criterion for Gamma operator
     * @param global local or global computation
     */
    HW( const Grid2d& g, double , double , double , double , bool);

    /**
     * @brief Returns phi and psi that belong to the last y in operator()
     *
     * In a multistep scheme this belongs to the point HEAD-1
     * @return phi[0] is the electron and phi[1] the generalized ion potential
     */
    const container& potential( ) const { return phi;}


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
    double capital_jot( ) {return jot_;}
    double capital_r() {return capitalR_;}
    /**
     * @brief Return the last integrated mass diffusion of operator() in a global computation
     *
     * @return int \nu \Delta (exp(y[0])-1)
     * @note undefined for a local computation
     */
    double flux( ) {return flux_;}
    /**
     * @brief Return the energy of the last field in operator() in a global computation
     *
     * @return integrated total energy in {ne, ni}
     * @note undefined for a local computation
     */
    double energy( ) {return energy_;}
    double zonal_flow_energy( ) {return uzf_;}
    /**
     * @brief Return the integrated energy diffusion of the last field in operator() in a global computation
     *
     * @return integrated total energy diffusion
     * @note undefined for a local computation
     */
    double energy_diffusion( ){ return ediff_;}
    double zonal_flow_diffusion() {return diff_;}

  private:
    const container& polarisation( const std::vector<container>& y);

    container chi, omega;

    container phi, phi_old, dyphi, lapphiM;
    std::vector<container> lapy, laplapy;

    //matrices and solvers
    ArakawaX< dg::CartesianGrid2d, Matrix, container> arakawa; 
    CG<container > pcg;
    PoloidalAverage<container, thrust::device_vector<int> > average;
    Elliptic<dg::CartesianGrid2d, Matrix, container> A, laplaceM;

    const container w2d, v2d, one;
    const double alpha;
    const double g;
    const double nu;
    const double eps_pol; 
    const bool mhw;

    double flux_, jot_, energy_, ediff_;
    double uzf_, capitalR_, diff_;

};

template< class Matrix, class container>
HW<Matrix, container>::HW( const Grid2d& grid, double alpha, double g, double nu, double eps_pol, bool mhw ): 
    chi( grid.size(), 0.), omega(chi), phi( chi), phi_old( chi), dyphi( chi),
    lapphiM(chi), lapy( 2, chi),  laplapy( lapy),
    A( grid, not_normed, centered), laplaceM( grid, normed, centered),
    arakawa( grid), 
    pcg( omega, omega.size()), 
    average( grid),
    w2d( create::weights(grid)), v2d( create::inv_weights(grid)), one( dg::evaluate(dg::one, grid)),
    alpha( alpha), g(g), nu( nu), eps_pol(eps_pol), mhw( mhw)
{

}

//computes and modifies expy!!
template<class M, class container>
const container& HW<M, container>::polarisation( const std::vector<container>& y)
{
    //extrapolate phi and gamma_n
    blas1::axpby( 2., phi, -1.,  phi_old);
    phi.swap( phi_old);
#ifdef DG_BENCHMARK
    Timer t; 
    t.tic();
#endif
    blas1::axpby( 1., y[1], -1., y[0], lapphiM); //n_i - n_e = omega
    blas2::symv( w2d, lapphiM, omega); 
    unsigned number = pcg( A, phi, omega, v2d, eps_pol);
    if( number == pcg.get_max())
        throw Fail( eps_pol);
#ifdef DG_BENCHMARK
    std::cout << "# of pcg iterations for phi \t"<< number <<"\t";
    t.toc();
    std::cout<< "took \t"<<t.diff()<<"s\n";
    double meanPhi = dg::blas2::dot( phi, w2d, one);
    std::cout << "Mean phi "<<meanPhi<<"\n";
#endif //DG_DEBUG
    return phi;
}

template< class M, class container>
void HW< M, container>::operator()( std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 2);
    assert( y.size() == yp.size());

    phi = polarisation( y);

    for( unsigned i=0; i<y.size(); i++)
        arakawa( y[i], phi, yp[i]);



    //compute derivatives
    blas2::gemv( arakawa.dy(), phi, dyphi);
    //gradient terms
    blas1::axpby( g, dyphi, 1., yp[0]);
    blas1::pointwiseDot( dyphi, lapphiM, omega);
    //blas1::axpby( -2.*0.05, omega, 1., yp[0]);
    blas1::axpby( g, dyphi, 1., yp[1]);
    blas1::axpby( -2.*0.05, omega, 1., yp[1]);
    //hw term
    blas1::axpby( 1., phi, 0., omega);
    blas1::axpby( -1., y[0], 1., omega);
    if( mhw) 
    {
        average( phi, chi);
        blas1::axpby( -1., chi, 1., omega );
        average( y[0], chi);
        blas1::axpby( 1., chi, 1., omega);
    }
    blas1::axpby( alpha, omega, 1., yp[0]);

    //add laplacians
    for( unsigned i=0; i<y.size(); i++)
    {
        blas2::gemv( laplaceM, y[i], lapy[i]);
        blas2::gemv( laplaceM, lapy[i], laplapy[i]);
       // blas1::axpby( -nu, laplapy[i], 1., yp[i]); //rescale 
    }

    double ue = 0.5*blas2::dot( y[0], w2d, y[0]);
    double ui = 0.5*blas2::dot( y[1], w2d, y[1]);
    double ei = 0.5*blas2::dot( phi, w2d, lapphiM);
    energy_ = ue + ui + ei;
    flux_ = g*blas2::dot( y[0], w2d, dyphi);
    jot_ = -alpha*blas2::dot( omega, w2d, omega);
    ediff_ = nu * blas2::dot( phi, w2d, laplapy[0]) -nu*blas2::dot( y[0], w2d, laplapy[0]) - nu * blas2::dot( phi, w2d, laplapy[1]);

     
    
    average( phi, chi);
    average( lapphiM, omega);
    uzf_ = 0.5*blas2::dot( chi, w2d, omega);
    blas1::pointwiseDot( dyphi, lapphiM, omega);
    blas2::symv( arakawa.dx(), omega, lapy[0]);
    average( lapy[0], omega);
    capitalR_ = blas2::dot( chi, w2d, omega);
    average( laplapy[0], omega);
    diff_ = nu * blas2::dot( phi, w2d, omega);
    average( laplapy[1], omega);
    diff_ += -nu * blas2::dot( phi, w2d, omega);

}


}//namespace dg
