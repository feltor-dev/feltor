#pragma once

#include "dg/algorithm.h"
#include "dg/poisson.h"
#include "parameters.h"
// #include "geometry_circ.h"
#include "dg/backend/average.cuh"

#ifdef DG_BENCHMARK
#include "dg/backend/timer.cuh"
#endif //DG_BENCHMARK
/*!@file

  Contains the solvers 
  */

namespace eule
{
///@addtogroup solver
///@{
/**
 * @brief Diffusive terms for Feltor solver
 *
 * @tparam Matrix The Matrix class
 * @tparam container The Vector class 
 * @tparam Preconditioner The Preconditioner class
 */

template<class Matrix, class container, class Preconditioner>
struct Rolkar
{
    template<class Grid2d>
    Rolkar( const Grid2d& g, eule::Parameters p):
        p(p),
        temp( dg::evaluate(dg::zero, g)),
        LaplacianM_perp ( g,g.bcx(),g.bcy(), dg::normed, dg::centered)
    {
    }
    void operator()( std::vector<container>& x, std::vector<container>& y)
    {
        /* x[0] := N_e - (bgamp+profamp)
           x[1] := N_i - (bgamp+profamp)
           x[2] := T_e - (bgamp+profamp)
           x[3] := T_i - (bgamp+profamp)

        */
        dg::blas1::axpby( 0., x, 0, y);
        for( unsigned i=0; i<4; i++)
        {
            //not linear any more (cannot be written as y = Ax)
            dg::blas2::gemv( LaplacianM_perp, x[i], temp);
            dg::blas2::gemv( LaplacianM_perp, temp, y[i]);
            dg::blas1::scal( y[i], -p.nu_perp);  //  nu_perp lapl_RZ (lapl_RZ N) 
        }


    }
    dg::Elliptic<Matrix, container, Preconditioner>& laplacianM() {return LaplacianM_perp;}
    const Preconditioner& weights(){return LaplacianM_perp.weights();}
    const Preconditioner& precond(){return LaplacianM_perp.precond();}
  private:
    const eule::Parameters p;
    container temp;    
    dg::Elliptic<Matrix, container, Preconditioner> LaplacianM_perp;

};

template< class Matrix, class container=thrust::device_vector<double>, class Preconditioner = thrust::device_vector<double> >
struct Feltor
{
    //typedef std::vector<container> Vector;
    typedef typename dg::VectorTraits<container>::value_type value_type;
    //typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    //typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    //typedef dg::DMatrix Matrix; //fastest device Matrix (does this conflict with 

    template<class Grid2d>
    Feltor( const Grid2d& g, eule::Parameters p);


    /**
     * @brief Returns phi and psi that belong to the last y in operator()
     *
     * In a multistep scheme this belongs to the point HEAD-1
     * @return phi[0] is the electron and phi[1] the generalized ion potential
     */
    const std::vector<container>& potential( ) const { return phi;}
    void initializene( const container& y, container& target);

    void operator()( std::vector<container>& y, std::vector<container>& yp);

    double mass( ) {return mass_;}
    double mass_diffusion( ) {return diff_;}
    double energy( ) {return energy_;}
    double coupling( ) {return coupling_;}

    std::vector<double> energy_vector( ) {return evec;}
    std::vector<container>& probe_vector( ) {return probevec;}

    double energy_diffusion( ){ return ediff_;}
    double radial_transport( ){ return gammanex_;}
    void energies( std::vector<container>& y);

  private:
    //extrapolates and solves for phi[1], then adds square velocity ( omega)
    container& compute_psi( container& potential);
    container& polarisation( const std::vector<container>& y); //solves polarisation equation

    container chi, omega, lambda; //!!Attention: chi and omega are helper variables and may be changed at any time and by any method!!
    container neavg,netilde,nedelta,lognedelta,phiavg,phitilde,phidelta,Niavg; //dont use them as helper
    const container binv;
    const container one;
    const Preconditioner w2d, v2d;
    std::vector<container> phi,phi2; // =(phi,psi_i), (0,chi_i)
    std::vector<container> npe, logn; 

    //matrices and solvers
    dg::Poisson< Matrix, container> poisson; 

    dg::Elliptic< Matrix, container, Preconditioner > pol,lapperp; 
    dg::Helmholtz< Matrix, container, Preconditioner > invgammaDIR,invgammaNU;

    dg::Invert<container> invert_pol,invert_invgamma;
    dg::PoloidalAverage<container, container > polavg;

    const eule::Parameters p;

    double mass_, energy_, diff_, ediff_,gammanex_,coupling_;
    std::vector<double> evec;
    //probe
    std::vector<container> probevec;
    const container Xprobe,Yprobe;
    Matrix probeinterp;
    container probevalue;
    
    dg::Grid1d<double> gy;
    const container w1d;
    const container oney;
    const container coox0,cooxlx,cooy;
    Matrix interpx0,interpxlx;
    container lh,rh,lhs,profne,profNi;

};     

template<class Matrix, class container, class P>
template<class Grid>
Feltor<Matrix, container, P>::Feltor( const Grid& g, eule::Parameters p): 
    chi( dg::evaluate( dg::one, g)), omega(chi),  lambda(chi), 
    neavg(chi),netilde(chi),nedelta(chi),lognedelta(chi),phiavg(chi),phitilde(chi),phidelta(chi),
    Niavg(chi),
    binv( dg::evaluate( dg::LinearX( p.mcv, 1.), g) ),
    one( dg::evaluate( dg::one, g)),    
    w2d( dg::create::weights(g)), v2d( dg::create::inv_weights(g)), 
    phi( 2, chi),phi2(2,chi), npe(phi), logn(phi),
    poisson(g, g.bcx(), g.bcy(), g.bcx(), g.bcy()), //first N/U then phi BCC
    pol(    g, g.bcx(), g.bcy(), dg::not_normed,          dg::centered), 
    lapperp ( g,g.bcx(), g.bcy(),     dg::normed,         dg::centered),
    invgammaDIR( g,g.bcx(), g.bcy(),-0.5*p.tau[1]*p.mu[1],dg::centered),
    invgammaNU(  g,g.bcx(), g.bcy(),-0.5*p.tau[1]*p.mu[1],dg::centered),
    invert_pol(      omega, omega.size(), p.eps_pol),
    invert_invgamma( omega, omega.size(), p.eps_gamma),
    polavg(g),
    p(p),
    evec(3),
    //probe
    probevec(2),
    Xprobe(1,p.lx*p.posX), //use blob position
    Yprobe(1,p.ly*p.posY),//use blob position
    probeinterp(dg::create::interpolation( Xprobe,  Yprobe,g, dg::NEU)),
    probevalue(1,0.0),
    lh( dg::evaluate(dg::TanhProfX(p.lx*p.solb,p.solw,-1.0,0.0,1.0),g)),
    rh( dg::evaluate(dg::TanhProfX(p.lx*p.solb,p.solw,1.0,0.0,1.0),g)), 
    lhs(dg::evaluate(dg::TanhProfX(p.lx*p.solb*0.7,p.solw,-1.0,0.0,1.0),g)),
    profne(dg::evaluate(dg::ExpProfX(p.nprofileamp, p.bgprofamp,p.ln),g)),
//     profne(dg::evaluate(dg::TanhProfX(p.lx*p.solb*0.7,p.solw,-1.0,p.nprofileamp, p.bgprofamp),g)),
    profNi(profne),
    //boundary integral terms
    gy(g.y0(),g.y1(),g.n(),g.Ny(),dg::PER),
    w1d( dg::create::weights(gy)),
    oney( dg::evaluate( dg::one, gy)), 
    coox0(dg::evaluate(dg::CONSTANT(0.0),gy)),
    cooxlx(dg::evaluate(dg::CONSTANT(p.lx),gy)),
    cooy(dg::evaluate(dg::coo1,gy)),
    interpx0(dg::create::interpolation( coox0,cooy, g )),  
    interpxlx(dg::create::interpolation(cooxlx,cooy, g))
{

}

template<class Matrix, class container, class P>
container& Feltor<Matrix, container, P>::polarisation( const std::vector<container>& y)
{
    dg::blas1::axpby( p.mu[1], y[1], 0, chi);      //chi =  \mu_i (n_i-(bgamp+profamp)) 
    dg::blas1::transform( chi, chi, dg::PLUS<>( p.mu[1]*(p.bgprofamp + p.nprofileamp))); //mu_i n_i
    dg::blas1::pointwiseDot( chi, binv, chi);
    dg::blas1::pointwiseDot( chi, binv, chi);       //(\mu_i n_i ) /B^2
    pol.set_chi( chi);
    invert_invgamma(invgammaNU,chi,y[1]); //chi= Gamma (Ni-(bgamp+profamp))    
    dg::blas1::axpby( -1., y[0], 1.,chi,chi);               //chi=  Gamma (n_i-(bgamp+profamp)) -(n_e-(bgamp+profamp))
    //= Gamma n_i - n_e
    unsigned number = invert_pol( pol, phi[0], chi);            //Gamma n_i -ne = -nabla chi nabla phi
        if(  number == invert_pol.get_max())
            throw dg::Fail( p.eps_pol);
    return phi[0];
}

template< class Matrix, class container, class P>
container& Feltor<Matrix,container, P>::compute_psi( container& potential)
{
    invert_invgamma(invgammaDIR,chi,potential);                 //chi  = Gamma phi
    poisson.variationRHS(potential, omega);
    dg::blas1::pointwiseDot( binv, omega, omega);
    dg::blas1::pointwiseDot( binv, omega, omega);
    dg::blas1::axpby( 1., chi, -0.5, omega,phi[1]);             //psi  Gamma phi - 0.5 u_E^2
    return phi[1];    
}

template<class Matrix, class container, class P>
void Feltor<Matrix, container, P>::initializene( const container& src, container& target)
{ 
    
    invert_invgamma(invgammaNU,target,src); //=ne-1 = Gamma_dagger (ni-1)    
    
}

template<class M, class V, class P>
void Feltor<M, V, P>::energies( std::vector<V>& y)
{
   
}


template<class Matrix, class container, class P>
void Feltor<Matrix, container, P>::operator()( std::vector<container>& y, std::vector<container>& yp)
{
    /* y[0] := N_e - (p.bgprofamp + p.nprofileamp)
       y[1] := N_i - (p.bgprofamp + p.nprofileamp)
       y[2] := T_e - (p.bgprofamp + p.nprofileamp)
       y[3] := T_i - (p.bgprofamp + p.nprofileamp)
    */
    dg::Timer t;
    t.tic();
    assert( y.size() == 2);
    assert( y.size() == yp.size());
    //compute phi via polarisation
    phi[0] = polarisation( y);
    phi[1] = compute_psi( phi[0]); //sets omega

    //transform compute n and logn and energies
    for(unsigned i=0; i<2; i++)
    {
        dg::blas1::transform( y[i], npe[i], dg::PLUS<>(+(p.bgprofamp + p.nprofileamp))); //npe = N+1
        dg::blas1::transform( npe[i], logn[i], dg::LN<value_type>());
    }
  
    t.toc();
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if(rank==0)
#endif 
    std::cout << "One rhs took "<<t.diff()<<"s\n";


}

///@}

} //namespace eule

