#pragma once

#include "dg/algorithm.h"
#include "parameters.h"
// #include "geometry_circ.h"
// #include "solovev/geometry.h"
#include "geometry_g.h"
#include "init.h"

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
    template<class Grid3d>
    Rolkar( const Grid3d& g, eule::Parameters p, solovev::GeomParameters gp):
        p(p),
        gp(gp),
        dampprof_( dg::evaluate( solovev::GaussianProfDamping( gp), g)),
        elliptic( g, dg::normed, dg::forward)

    {
        container bfield = dg::evaluate( solovev::bR( gp.R_0, gp.I_0),g);
        elliptic.set_x( bfield);
        bfield = dg::evaluate( solovev::bZ( gp.R_0, gp.I_0),g);
        elliptic.set_y( bfield);
        bfield = dg::evaluate( solovev::bPhi( gp.R_0, gp.I_0),g);
        elliptic.set_z( bfield);
    }
    void operator()( std::vector<container>& x, std::vector<container>& y)
    {
          dg::blas1::axpby( 0., x, 0, y);
      
        if (p.p_diff ==4)    {
        dg::blas2::gemv( elliptic, x[0], y[0]); //lap is negative
        dg::blas1::scal(y[0], -p.nu_parallel );  
        }
    }
    const container& damping(){return dampprof_;}
    const Preconditioner& weights(){return elliptic.weights();}
    const Preconditioner& precond(){return elliptic.precond();}
  private:
    const eule::Parameters p;
    const solovev::GeomParameters gp;
    const container dampprof_;
  
    dg::GeneralElliptic<Matrix, container, Preconditioner> elliptic;

};

template< class Matrix, class container=thrust::device_vector<double>, class Preconditioner = thrust::device_vector<double> >
struct Feltor
{
    //typedef std::vector<container> Vector;
    typedef typename dg::VectorTraits<container>::value_type value_type;
    //typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    //typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    //typedef dg::DMatrix Matrix; //fastest device Matrix (does this conflict with 

    template<class Grid3d>
    Feltor( const Grid3d& g, eule::Parameters p,solovev::GeomParameters gp);

    dg::DZ<Matrix, container> dz(){return dzNU_;}

    void operator()( std::vector<container>& y, std::vector<container>& yp);

    double mass( ) {return mass_;}
    double mass_diffusion( ) {return diff_;}
    double energy( ) {return energy_;}
    std::vector<double> energy_vector( ) {return evec;}
    double energy_diffusion( ){ return ediff_;}

    void energies( std::vector<container>& y);

  private:


    container chi, omega, lambda,tmo; //!!Attention: chi and omega are helper variables and may be changed at any time and by any method!!

    const container binv, gradlnB,pupil;
    const container  one;
    const Preconditioner w3d, v3d;

    //matrices and solvers
    dg::DZ<Matrix, container> dzNU_;

    dg::Elliptic< Matrix, container, Preconditioner > lapperp; 
//     dg::GeneralElliptic<Matrix, container, Preconditioner> elliptic;

    const eule::Parameters p;
    const solovev::GeomParameters gp;

    double mass_, energy_, diff_, ediff_;
    std::vector<double> evec;

};

template<class Matrix, class container, class P>
template<class Grid>
Feltor<Matrix, container, P>::Feltor( const Grid& g, eule::Parameters p, solovev::GeomParameters gp): 
    chi( dg::evaluate( dg::one, g)), omega(chi),  lambda(chi), tmo(chi),
    binv( dg::evaluate(solovev::Field(gp) , g) ),
    gradlnB( dg::evaluate(solovev::GradLnB(gp) , g)),
//     pupil(dg::evaluate( solovev::Pupil( gp), g)),
    pupil(dg::evaluate( solovev::GaussianProfDamping(gp ), g)),    //be aware of actual function!
    one( dg::evaluate( dg::one, g)),    
    w3d( dg::create::weights(g)), v3d( dg::create::inv_weights(g)),      
    dzNU_(solovev::Field(gp), g, 2.*M_PI/(double)p.Nz, gp.rk4eps,solovev::PsiLimiter(gp), g.bcx()),
    lapperp ( g,g.bcx(), g.bcy(),     dg::normed,         dg::centered),
//     elliptic( g, dg::not_normed, dg::forward),    
    p(p),
    gp(gp),
    evec(1)
{ 
//     container bfield = dg::evaluate( solovev::bR( gp.R_0, gp.I_0),g);
//     elliptic.set_x( bfield);
//     bfield = dg::evaluate( solovev::bZ( gp.R_0, gp.I_0),g);
//     elliptic.set_y( bfield);
//     bfield = dg::evaluate( solovev::bPhi( gp.R_0, gp.I_0),g);
//     elliptic.set_z( bfield);
    
}



template<class M, class V, class P>
void Feltor<M, V, P>::energies( std::vector<V>& y)
{
    double S[1]    = {0.0};    
    double Dpar[1] = {0.0};
    double Dperp[1] = {0.0};
    S[0]    = dg::blas2::dot( one, w3d, y[0]);
    mass_ = S[0];
    energy_ = S[0]; 
    evec[0] = S[0]; 
//             dg::blas1::transform( y[0],tmo, dg::PLUS<>(-1)); //npe = N+1

    //perp energy
    if (p.p_diffperp==0)    {
        dg::blas2::gemv( lapperp, y[0], lambda);
        Dperp[0] = -p.nu_perp*dg::blas2::dot(one, w3d, lambda); 
    }
    if (p.p_diffperp==1)    {
        dg::blas2::gemv( lapperp, y[0], lambda);
        dg::blas2::gemv( lapperp, lambda, omega); //hyper
        Dperp[0] = -p.nu_perp*dg::blas2::dot(one, w3d, omega); //hyper 
    }
    if (p.p_torlim == 1)  {
         dzNU_.set_boundaries( p.bc, 0, 0); 
    }
    if (p.p_diff ==0)    {
    //     (A) adjoint
//         dzNU_( y[0], omega); 
//         dzNU_.centeredT(omega,lambda);
        dzNU_.forward( y[0], omega); 
        dzNU_.forwardT(omega,lambda);
        dg::blas1::axpby( 0.5, lambda, 0.,chi,chi); 

        dzNU_.backward( y[0], omega); 
        dzNU_.backwardT(omega,lambda);
        dg::blas1::axpby( 0.5, lambda, 1., chi,chi); 
        Dpar[0]= p.nu_parallel*dg::blas2::dot(one, w3d, chi);
    }  
    if (p.p_diff ==1)    {
        // (B) nonadjoint
        dzNU_( binv, lambda); //gradpar 1/B
        dg::blas1::pointwiseDivide(lambda,  binv, lambda); //-dz lnB
        dzNU_(y[0],omega); //dz T
        dg::blas1::pointwiseDot(omega, lambda, omega);            // -dz lnB dz T
        dzNU_.dzz(y[0],lambda);                                          //dz^2 T 
        dg::blas1::axpby( 1., omega,  1.,lambda );    
        Dpar[0]= p.nu_parallel*dg::blas2::dot(one, w3d, lambda);  
    }
    if (p.p_diff ==2)    {
        // (C) oldnonadjoint
        dzNU_.dzz(y[0],omega);                                          //dz^2 N 
        dzNU_(y[0],lambda);       
        dg::blas1::pointwiseDot(gradlnB, lambda, lambda);            // dz lnB dz N
        dg::blas1::axpby( 1., omega, -1., lambda);       
        Dpar[0]= p.nu_parallel*dg::blas2::dot(one, w3d, lambda);  
    }
    if (p.p_diff ==3)    {
        // (D) nonadjoint with direct method
        dzNU_.forward( y[0], omega); 
        dzNU_.forwardTD(omega,lambda);
        dg::blas1::axpby( 0.5, lambda, 0.,chi,chi); 

        dzNU_.backward( y[0], omega); 
        dzNU_.backwardTD(omega,lambda);
        dg::blas1::axpby( 0.5, lambda, 1., chi,chi); 
        Dpar[0]= p.nu_parallel*dg::blas2::dot(one, w3d, chi); 
    }
    //Compute rhs of energy theorem
    ediff_= Dpar[0]+Dperp[0];
}


//do not overwrite y
template<class Matrix, class container, class P>
void Feltor<Matrix, container, P>::operator()( std::vector<container>& y, std::vector<container>& yp)
{
    /* y[0] := T - 1 or T
    */
    dg::Timer t;
    t.tic();
    assert( y.size() == 1);
    assert( y.size() == yp.size());
    if (p.p_diffperp==0) {
        dg::blas2::gemv( lapperp, y[0], omega); //lap is negative
        dg::blas1::axpby( -p.nu_perp, omega, 0., yp[0]);  
    }
    if (p.p_diffperp==1) {
        dg::blas2::gemv( lapperp, y[0], omega); //lap is negative
        dg::blas2::gemv( lapperp, omega, lambda); //hyper
        dg::blas1::axpby( -p.nu_perp, lambda, 0., yp[0]);  //hyper 
    }   

    if (p.p_torlim == 1)  {
         dzNU_.set_boundaries( p.bc, 0, 0); 
    }   
//-----------------------parallel adv------------------------
    if (p.p_adv ==0)  {
        // (A) adjoint
        //U=v_parallel gradlnB
    //     dg::blas1::pointwiseDot(y[0], gradlnB, lambda);
    //     dzNU_.centeredT(lambda,omega);    
    //     dg::blas1::axpby( p.nu_parallel, omega, 1., yp[0]); 

        //U=1.  
        //centered
        dg::blas1::pointwiseDot(y[0],pupil,lambda);    //U*T  
        dzNU_.centeredT(lambda,omega);    // dzT UT
//         dzNU_.backwardT(lambda,omega);
//         dzNU_.forwardT(lambda,omega);   
        dg::blas1::axpby( -1.0, omega, 1., yp[0]); //dzT (UT)

        //corr(1): div(UB) 
//         dg::blas1::pointwiseDivide(pupil,binv,omega); //= U B
//         dzNU_.centeredT(omega,lambda);     //div UB
//         dg::blas1::axpby( 1.0, lambda, 1., yp[0]); //+div UB
        
        //corr(2): div(B) 
//         dg::blas1::pointwiseDivide(one,binv,omega); //= U B
//         dzNU_.centeredT(omega,lambda);     //div UB
//         dg::blas1::axpby( 1.0, lambda, 1., yp[0]); //+div UB
        
        //corr(3): UT/B divB 
    //     dg::blas1::pointwiseDivide(one,binv,omega); //=  B
    //     dzNU_.centeredT(omega,lambda);     //div B
    //     dg::blas1::pointwiseDot(y[0],binv,omega); //T/B
    //     dg::blas1::pointwiseDot(omega,pupil,omega); // U T/B
    //     dg::blas1::pointwiseDot(omega,lambda,lambda); //  UT/B divB
    //     dg::blas1::axpby( 1.0, lambda, 1., yp[0]); //+ UT/B div UB

        //corr(4): U  divB
//         dg::blas1::pointwiseDivide(one,binv,omega); //=  B
//         dzNU_.centeredT(omega,lambda);     //div B
//         dg::blas1::pointwiseDot(pupil,lambda,lambda); //  U divB
//         dg::blas1::axpby( 1.0, lambda, 1., yp[0]); //+ U div UB
    }
    if (p.p_adv ==1)    {
        // (B) nonadjoint U=1
        dg::blas1::pointwiseDot(y[0],pupil,lambda);    //UT
        dzNU_(lambda,omega);    //Dz UT
        dg::blas1::axpby( -1.0, omega, 1., yp[0]); //-  dz U T
        dzNU_( binv, lambda); //gradpar 1/B
        dg::blas1::pointwiseDivide(lambda,  binv, lambda); //-dz lnB  
        dg::blas1::pointwiseDot(y[0],pupil,omega);    //=  U T  
        dg::blas1::pointwiseDot(omega,  lambda, omega); //-U T dz lnB  
        dg::blas1::axpby( -1.0, omega, 1., yp[0]); //UT dzlnB
    }
    if (p.p_adv ==2)    {
        // (C) oldnonadjoint U=1
        dg::blas1::pointwiseDot(y[0],pupil,lambda);    // UT
        dzNU_(lambda,omega);    //  dz U T
        dg::blas1::axpby( -1.0, omega, 1., yp[0]); //-  dz U T
        dg::blas1::pointwiseDot(lambda,  gradlnB, omega); //U T dz lnB  
        dg::blas1::axpby( 1.0, omega, 1., yp[0]); //UT dzlnB
    }
//-----------------------parallel dissi------------------------
    if (p.p_diff ==0)    {
//         centered
//         dzNU_( y[0], omega); 
//         dzNU_.centeredT(omega,lambda);
//         dg::blas1::axpby( p.nu_parallel, lambda, 1., yp[0]); 

        //forward, backward (stegi)
        dzNU_.forward( y[0], omega); 
        dzNU_.forwardT(omega,lambda);
        dg::blas1::axpby( 0.5*p.nu_parallel, lambda, 1., yp[0]); 

        dzNU_.backward( y[0], omega); 
        dzNU_.backwardT(omega,lambda);
        dg::blas1::axpby( 0.5*p.nu_parallel, lambda, 1., yp[0]); 
        //with jump
//        dzNU_.symv(y[0],lambda);
//        dg::blas1::pointwiseDot(v3d,lambda,lambda);
//        dg::blas1::axpby( p.nu_parallel, lambda, 1., yp[0]); 

    }
    if (p.p_diff ==1)    {
        // (B) nonadjoint
        dzNU_( binv, lambda); //gradpar 1/B
        dg::blas1::pointwiseDivide(lambda,  binv, lambda); //-dz lnB
        dzNU_(y[0],omega); //dz T
        dg::blas1::pointwiseDot(omega, lambda, omega);            //- dz lnB dz T
        dg::blas1::axpby(p.nu_parallel, omega, 1., yp[0]);    
        dzNU_.dzz(y[0],omega);                                          //dz^2 T 
        dg::blas1::axpby( p.nu_parallel, omega, 1., yp[0]);
    }
    if (p.p_diff ==2)    {
        // (C) oldnonadjoint
        dzNU_.dzz(y[0],omega);                                          //dz^2 T 
        dg::blas1::axpby( p.nu_parallel, omega, 1., yp[0]);       
        dzNU_(y[0],lambda);       
        dg::blas1::pointwiseDot(gradlnB, lambda, omega);            // dz lnB dz T
        dg::blas1::axpby(-p.nu_parallel, omega, 1., yp[0]);    
    }
    if (p.p_diff ==3)    {
        // (D) nonadjoint with direct method
        dzNU_.forward( y[0], omega); 
        dzNU_.forwardTD(omega,lambda);
        dg::blas1::axpby( 0.5*p.nu_parallel, lambda, 1., yp[0]); 

        dzNU_.backward( y[0], omega); 
        dzNU_.backwardTD(omega,lambda);
        dg::blas1::axpby( 0.5*p.nu_parallel, lambda, 1., yp[0]); 
    }
//     if (p.p_diff ==4)    {
//         dg::blas2::gemv( elliptic, y[0], omega); //lap is negative
//         dg::blas1::axpby( -p.nu_parallel, omega, 1., yp[0]);  
//     }
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

