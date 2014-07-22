#pragma once

#include "dg/xspacelib.cuh"
#include "dg/cg.cuh"
#include "dg/dz.cuh"
#include "dg/gamma.cuh"

#include "parameters.h"
// #include "geometry_circ.h"
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
    Rolkar( const dg::Grid3d<double>& g, Parameters p, solovev::GeomParameters gp):
        p(p),
        gp(gp),
        w3d_( dg::create::w3d(g)), v3d_(dg::create::v3d(g)),
        //w3d( 4, &w3d_), v3d( 4, &v3d_), 
        temp( g.size()),
        pupil_( dg::evaluate( solovev::Pupil( gp), g))
    {
        LaplacianM_perp = dg::create::laplacianM_perp( g, dg::normed, dg::XSPACE);
        //LaplacianM_para = dg::create::laplacianM_parallel( g, dg::PER);
    }
    void operator()( const std::vector<container>& x, std::vector<container>& y)
    {
        for( unsigned i=0; i<x.size(); i++)
        {
            dg::blas2::gemv( LaplacianM_perp, x[i], temp);
            dg::blas2::gemv( LaplacianM_perp, temp, y[i]);
            dg::blas1::axpby( -p.nu_perp, y[i], 0., y[i]);
        }
        //add parallel resistivity
        std::vector<container>  expy(2);
        expy[0].resize( x[0].size()), expy[1].resize( x[1].size());
        container chi( x[0].size()), omega( x[0].size());
        for( unsigned i=0; i<2; i++)
            thrust::transform( x[i].begin(), x[i].end(), expy[i].begin(), dg::EXP<double>());
        dg::blas1::pointwiseDot( expy[0], x[2], omega);
        dg::blas1::pointwiseDot( expy[1], x[3], chi);
        dg::blas1::axpby( -1., omega, 1., chi); //-N_eU_e + N_iU_i
        dg::blas1::pointwiseDivide( chi, expy[0], omega);//J_par/N_e
        dg::blas1::pointwiseDivide( chi, expy[1], chi); //J_par/N_i

        dg::blas1::axpby( -p.c/p.mu[0]/p.eps_hat, omega, 1., y[2]);
        dg::blas1::axpby( -p.c/p.mu[1]/p.eps_hat, chi, 1., y[3]);
        //cut contributions to boundary 
        for( unsigned i=0; i<y.size(); i++)
            dg::blas1::pointwiseDot( pupil_, y[i], y[i]);
    }
    const dg::DMatrix& laplacianM()const {return LaplacianM_perp;}
    //const std::vector<const container*>& weights(){return w3d;}
    //const std::vector<const container*>& precond(){return v3d;}
    const container& weights(){return w3d_;}
    const container& precond(){return v3d_;}
    const container& iris(){return pupil_;}

  private:
    void divide( const container& zaehler, const container& nenner, container& result)
    {
        thrust::transform( zaehler.begin(), zaehler.end(), nenner.begin(), result.begin(), 
                thrust::divides< typename container::value_type>());
    }
    const Parameters p;
    const solovev::GeomParameters gp;
    const container w3d_, v3d_;
    const std::vector<const container*> w3d, v3d;
    container temp;
    const container pupil_;
    dg::DMatrix LaplacianM_perp;
    //dg::DMatrix LaplacianM_para;
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

    Feltor( const dg::Grid3d<value_type>& g, Parameters p,solovev::GeomParameters gp);

    void exp( const std::vector<container>& src, std::vector<container>& dst, unsigned);

    void log( const std::vector<container>& src, std::vector<container>& dst, unsigned);

    /**
     * @brief Returns phi and psi that belong to the last y in operator()
     *
     * In a multistep scheme this belongs to the point HEAD-1
     * @return phi[0] is the electron and phi[1] the generalized ion potential
     */
    const std::vector<container>& potential( ) const { return phi;}


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
    void curve( const container& y, container& target);
    //use chi and omega as helpers to compute square velocity in omega
    const container& compute_vesqr( const container& potential);
    //extrapolates and solves for phi[1], then adds square velocity ( omega)
    const container& compute_psi( const container& potential);
    const container& polarisation( const std::vector<container>& y);

    container chi, omega;
    const container binv, curvR, curvZ, gradlnB;
    const container iris, source, damping;

    std::vector<container> phi, curvphi, dzphi, expy;
    std::vector<container> dzy, curvy; 

    //matrices and solvers
    Matrix A; 
    dg::DZ<container> dz;
    dg::ArakawaX< dg::DMatrix, container>    arakawa; 
    //dg::Polarisation2dX< thrust::host_vector<value_type> > pol; //note the host vector
    dg::Polarisation2dX< container, dg::DMatrix > pol; //note the host vector
    dg::Invert<container>       invert_pol;

    const container w3d, v3d, one;
    const Parameters p;
    const solovev::GeomParameters gp;

    double mass_, energy_, diff_, ediff_;

};

// template< class container>
// Feltor< container>::Feltor( const dg::Grid3d<value_type>& g, Parameters p, GeomParam gp): 
//     chi( g.size(), 0.), omega(chi),
//     binv( dg::evaluate(Field(geop) , g) ),
//     curvR( dg::evaluate( eule::CurvatureR(p.R_0, p.I_0), g)),
//     curvZ( dg::evaluate( eule::CurvatureZ(p.R_0, p.I_0), g)),
//     gradlnB( dg::evaluate( eule::GradLnB(p.R_0, p.I_0) , g)),
//     iris( dg::evaluate( eule::Pupil( p.R_0, p.a, p.b), g)),
//     source( dg::evaluate( dg::Gaussian( p.R_0, 0, p.b, p.b, p.amp_source, 0), g)),
//     damping( dg::evaluate( eule::Damping( p.R_0, p.a, p.b, p.damping_width, p.damping_strength ), g)), 
//     phi( 2, chi), curvphi( phi), dzphi(phi), expy(phi),  
//     dzy( 4, chi), curvy(dzy),
//     A (dg::create::laplacianM_perp( g, dg::not_normed, dg::XSPACE, dg::symmetric)),
//     dz( eule::Field(p.R_0, p.I_0), g),
//     arakawa( g), 
//     pol(     g), 
//     invert_pol( omega, omega.size(), p.eps_pol), 
//     w3d( dg::create::w3d(g)), v3d( dg::create::v3d(g)), one( g.size(), 1.),
//     p(p)
// {
// }
template< class container>
Feltor< container>::Feltor( const dg::Grid3d<value_type>& g, Parameters p, solovev::GeomParameters gp): 
    chi( g.size(), 0.), omega(chi),
    binv( dg::evaluate(solovev::Field(gp) , g) ),
    curvR( dg::evaluate( solovev::CurvatureR(gp), g)),
    curvZ( dg::evaluate(solovev::CurvatureZ(gp), g)),
    gradlnB( dg::evaluate(solovev::GradLnB(gp) , g)),
    iris( dg::evaluate( solovev::Pupil( gp), g)),
    source( dg::evaluate( dg::Gaussian( p.R_0, 0, p.b, p.b, p.amp_source, 0), g)),
    damping( dg::evaluate( solovev::Damping(gp ), g)), 
    phi( 2, chi), curvphi( phi), dzphi(phi), expy(phi),  
    dzy( 4, chi), curvy(dzy),
    A (dg::create::laplacianM_perp( g, dg::not_normed, dg::XSPACE, dg::symmetric)),
    dz(solovev::Field(gp), g),
    arakawa( g), 
    pol(     g), 
    invert_pol( omega, omega.size(), p.eps_pol), 
    w3d( dg::create::w3d(g)), v3d( dg::create::v3d(g)), one( g.size(), 1.),
    p(p),
    gp(gp)
{
}
template< class container>
const container& Feltor<container>::compute_vesqr( const container& potential)
{
    arakawa.bracketS( potential, potential, chi);
    dg::blas1::pointwiseDot( binv, binv, omega);
    dg::blas1::pointwiseDot( chi, omega, omega);
    return omega;
}
template< class container>
const container& Feltor<container>::compute_psi( const container& potential)
{
    dg::blas1::axpby( 1., potential, -0.5, compute_vesqr( potential), phi[1]);
    return phi[1];
}


//computes and modifies expy!!
template<class container>
const container& Feltor< container>::polarisation( const std::vector<container>& y)
{
#ifdef DG_BENCHMARK
    dg::Timer t; 
    t.tic();
#endif
    //compute chi and polarisation
    exp( y, expy, 2);
    dg::blas1::axpby( 1., expy[1], 0., chi); //\chi = a_i \mu_i n_i
    dg::blas1::pointwiseDot( chi, binv, chi);
    dg::blas1::pointwiseDot( chi, binv, chi); //chi/= B^2
    //A = pol.create( chi);
    pol.set_chi( chi);
    thrust::transform( expy[0].begin(), expy[0].end(), expy[0].begin(), dg::PLUS<double>(-1)); //n_e -1
    thrust::transform( expy[1].begin(), expy[1].end(), omega.begin(), dg::PLUS<double>(-1)); //n_i -1
#ifdef DG_BENCHMARK
    t.toc();
    //std::cout<< "Polarisation assembly took "<<t.diff()<<"s\n";
#endif 
    dg::blas1::axpby( -1., expy[0], 1., omega); //n_i-n_e
    //unsigned number = invert_pol( A, phi[0], omega, w3d, v3d);
    unsigned number = invert_pol( pol, phi[0], omega, w3d, v3d);
    if( number == invert_pol.get_max())
        throw Fail( p.eps_pol);
    return phi[0];
}

template< class container>
void Feltor< container>::operator()( const std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 4);
    assert( y.size() == yp.size());

    phi[0] = polarisation( y);
    phi[1] = compute_psi( phi[0]);

    //update energetics, 2% of total time
    exp( y, expy, 2);
    mass_ = dg::blas2::dot( one, w3d, expy[0] ); //take real ion density which is electron density!!
    double Ue = p.tau[0]*dg::blas2::dot( y[0], w3d, expy[0]);
    double Ui = p.tau[1]*dg::blas2::dot( y[1], w3d, expy[1]);
    double Uphi = 0.5*p.mu[1]*dg::blas2::dot( expy[1], w3d, omega); 
    dg::blas1::pointwiseDot( y[2], y[2], omega);
    double Upare = -0.5*p.mu[0]*dg::blas2::dot( expy[0], w3d, omega); 
    dg::blas1::pointwiseDot( y[3], y[3], omega);
    double Upari =  0.5*p.mu[1]*dg::blas2::dot( expy[1], w3d, omega); 
    energy_ = Ue + Ui + Uphi + Upare + Upari;

    for( unsigned i=0; i<2; i++)
    {

        arakawa( y[i], phi[i], yp[i]);
        arakawa( y[i+2], phi[i], yp[i+2]);
        dg::blas1::pointwiseDot( yp[i], binv, yp[i]);
        dg::blas1::pointwiseDot( yp[2+i], binv, yp[2+i]);

        //compute parallel derivatives
        dz(y[i], dzy[i]);
        dz(phi[i], dzphi[i]);
        dz(y[2+i], dzy[2+i]);

        //parallel advection terms
        dg::blas1::pointwiseDot(y[2+i], dzy[i], omega); //Udz lnN 
        dg::blas1::axpby( -1., omega, 1., yp[i]); //-Udz lnN
        dg::blas1::axpby( -1., dzy[2+i], 1., yp[i]); //-dz U
        dg::blas1::pointwiseDot(y[2+i], gradlnB, omega);  // U dz ln B
        dg::blas1::axpby( 1., omega, 1., yp[i]); 
        dg::blas1::pointwiseDot(y[2+i], dzy[2+i], omega); // U dz U
        dg::blas1::axpby( -1., omega, 1., yp[2+i]);

        //parallel force terms
        dg::blas1::axpby( -p.tau[i]/p.mu[i]/p.eps_hat, dzy[i], 1., yp[2+i]);
        dg::blas1::axpby( -1./p.mu[i]/p.eps_hat, dzphi[i], 1., yp[2+i]);

        //curvature terms
        curve( y[i], curvy[i]);
        curve( y[2+i], curvy[2+i]);
        curve( phi[i], curvphi[i]);

        dg::blas1::pointwiseDot( y[2+i], curvy[2+i], omega); //UK(U)
        dg::blas1::pointwiseDot( y[2+i], omega, chi); //U^2K(U)
        dg::blas1::axpby( -p.mu[i]*p.eps_hat, omega, 1., yp[i]); //-mu UK(U)
        dg::blas1::axpby( -0.5*p.mu[i]*p.eps_hat, chi, 1., yp[2+i]); //-0.5mu U^2K(U)

        dg::blas1::pointwiseDot( y[2+i], curvy[i], omega);//UK(ln N)
        dg::blas1::pointwiseDot( y[2+i], omega, chi); //U^2K(ln N)
        dg::blas1::axpby( -p.tau[i], omega, 1., yp[2+i]);//-tau UK(ln N)
        dg::blas1::axpby( -0.5*p.mu[i]*p.eps_hat, chi, 1., yp[i]); //-0.5mu U^2K(ln N)

        dg::blas1::axpby( -p.tau[i], curvy[i], 1., yp[i]); //-tau K(lnN)
        dg::blas1::axpby( -2.*p.tau[i], curvy[2+i], 1., yp[2+i]); //-2 tau K(U)
        dg::blas1::axpby( -1., curvphi[i], 1., yp[i]); //-K(psi)

        dg::blas1::pointwiseDot( y[2+i], curvphi[i], omega); //UK(psi)
        dg::blas1::axpby( -0.5, omega, 1., yp[2+i]); //-0.5 UK(psi)


    }
    //add parallel resistivity
    //dg::blas1::pointwiseDot( expy[0], y[2], omega);
    //dg::blas1::pointwiseDot( expy[1], y[3], chi);
    //dg::blas1::axpby( -1., omega, 1., chi); //-N_eU_e + N_iU_i
    //dg::blas1::pointwiseDivide( chi, expy[0], omega);//J_par/N_e
    //dg::blas1::pointwiseDivide( chi, expy[1], chi); //J_par/N_i

    //dg::blas1::axpby( -p.c/p.mu[0]/p.eps_hat, omega, 1., yp[2]);
    //dg::blas1::axpby( -p.c/p.mu[1]/p.eps_hat, chi, 1., yp[3]);
    //add parallel diffusion
    for( unsigned i=0; i<4; i++)
    {
        dz(dzy[i], omega);
        dg::blas1::axpby( p.nu_parallel, omega, 1., yp[i]);
    }
    //add particle source
    for( unsigned i=0; i<2; i++)
    {
        dg::blas1::pointwiseDivide( source, expy[i], omega);
        dg::blas1::axpby( 1., omega, 1, yp[i]  );
    }
    //cut boundary terms 
    for( unsigned i=0; i<2; i++)
    {
        //dg::blas1::pointwiseDivide( damping, expy[i], omega); 
        //dg::blas1::axpby( -1., omega, 1., yp[i]);
        //dg::blas1::pointwiseDot( damping, y[i], omega); 
        //dg::blas1::axpby( -1., omega, 1., yp[i]);
    }
    for( unsigned i=0; i<4; i++)
    {
        dg::blas1::pointwiseDot( damping, yp[i], yp[i]); 
        dg::blas1::pointwiseDot( iris, yp[i], yp[i]);
    }
}

template< class container>
void Feltor< container>::curve( const container& src, container& target)
{
    dg::blas2::gemv( arakawa.dx(), src, target);
    dg::blas2::gemv( arakawa.dy(), src, omega);
    dg::blas1::pointwiseDot( curvR, target, target);
    dg::blas1::pointwiseDot( curvZ, omega, omega);
    dg::blas1::axpby( 1., omega, 1., target );
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


} //namespace eule
