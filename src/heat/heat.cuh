#pragma once

#include "dg/algorithm.h"
#include "parameters.h"
#include "geometries/geometries.h"

#ifdef DG_BENCHMARK
#include "dg/backend/timer.cuh"
#endif //DG_BENCHMARK
/*!@file

  Contains the solvers 
  */

namespace heat
{
///@addtogroup solver
///@{
template< class Geometry, class IMatrix, class Matrix, class container>
struct Implicit
{
    Implicit( const Geometry& g, Parameters p, dg::geo::solovev::Parameters gp):
        p(p),
        gp(gp),
        dsNU_(dg::geo::createSolovevField( gp), g, g.bcx(), g.bcy(),
              dg::geo::PsiLimiter( dg::geo::solovev::Psip(gp), gp.psipmaxlim), 
              dg::normed, dg::forward,
              gp.rk4eps, 1,1, true, true), 
        elliptic( g, dg::normed, dg::forward)
    {
        using namespace dg::geo::solovev;
        dg::geo::TokamakMagneticField c = dg::geo::createSolovevField(gp);
        container bfield = dg::pullback( dg::geo::FieldR(c),g);
        elliptic.set_x( bfield);
        bfield = dg::pullback( dg::geo::FieldZ(c),g);
        elliptic.set_y( bfield);
        bfield = dg::pullback( dg::geo::FieldP(c),g);
        elliptic.set_z( bfield);

        dg::blas1::transfer( dg::pullback( dg::geo::GaussianProfDamping(c.psip(), gp.psipmax, gp.alpha), g), dampprof_);
    }
    void operator()( const std::vector<container>& x, std::vector<container>& y)
    {
        dg::blas1::scal( y, 0.);
        if (p.p_diff ==0)    {
          dg::blas2::gemv( dsNU_, x[0], y[0]); 
          dg::blas1::scal(y[0], p.nu_parallel );  
       }
        if (p.p_diff ==4)    {
            dg::blas2::gemv( elliptic, x[0], y[0]); //lapd is negative
            dg::blas1::scal(y[0], -p.nu_parallel );  
        }
    }
    const container& damping(){return dampprof_;}
    const container& weights(){return elliptic.weights();}
    const container& inv_weights(){return elliptic.inv_weights();}
    const container& precond(){return elliptic.precond();}
  private:
    const heat::Parameters p;
    const dg::geo::solovev::Parameters gp;
    container dampprof_;
    dg::geo::DS<Geometry, IMatrix, Matrix, container> dsNU_;
    dg::GeneralEllipticSym<Geometry, Matrix, container> elliptic;

};

template< class Geometry, class IMatrix, class Matrix, class container >
struct Explicit
{
    Explicit( const Geometry& g, heat::Parameters p, dg::geo::solovev::Parameters gp);

    const dg::geo::DS<Geometry,IMatrix,Matrix,container>& ds(){return dsNU_;}

    void operator()( const std::vector<container>& y, std::vector<container>& yp);

    double mass( ) {return mass_;}
    double mass_diffusion( ) {return diff_;}
    double energy( ) {return energy_;}
    std::vector<double> energy_vector( ) {return evec;}
    double energy_diffusion( ){ return ediff_;}
    void energies( std::vector<container>& y);

  private:


    container chi, omega, lambda,tmo; //!!Attention: chi and omega are helper variables and may be changed at any time and by any method!!
    container m_invB, gradlnB;
//     ,pupil;
    const container  one;
    const container w3d, v3d;

    //matrices and solvers
    dg::geo::DS<Geometry,IMatrix,Matrix,container> dsDIR_, dsNU_;

    const heat::Parameters p;
    const dg::geo::solovev::Parameters gp;
    std::vector<double> evec;

    double mass_, energy_, diff_, ediff_;

};

template<class Geometry, class IMatrix, class Matrix, class container>
Explicit<Geometry,IMatrix,Matrix,container>::Explicit( const Geometry& g, heat::Parameters p, dg::geo::solovev::Parameters gp): 
    chi( dg::evaluate( dg::one, g)), omega(chi),  lambda(chi), tmo(chi),
    one( dg::evaluate( dg::one, g)),    
    w3d( dg::create::volume(g)), v3d( dg::create::inv_volume(g)),      
    dsDIR_( dg::geo::createSolovevField(gp), g, dg::DIR,dg::DIR,dg::geo::PsiLimiter( dg::geo::solovev::Psip(gp), gp.psipmaxlim), dg::normed, dg::centered), 
    dsNU_( dg::geo::createSolovevField(gp), g, dg::NEU,dg::NEU,dg::geo::PsiLimiter( dg::geo::solovev::Psip(gp), gp.psipmaxlim), dg::normed, dg::centered), 
    p(p),
    gp(gp),
    evec(1)
{
    //////////////////////////////init fields /////////////////////
    dg::geo::TokamakMagneticField mf = dg::geo::createSolovevField(gp);
    dg::blas1::transfer(  dg::pullback(dg::geo::InvB(mf),    g), m_invB);
    dg::blas1::transfer(  dg::pullback(dg::geo::GradLnB(mf), g), gradlnB);
}



template<class G,class I, class M, class V>
void Explicit<G,I,M,V>::energies( std::vector<V>& y)
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

//     if (p.p_diffperp==0)    {
//         dg::blas2::gemv( lapperp, y[0], lambda);
//         Dperp[0] = -p.nu_perp*dg::blas2::dot(one, w3d, lambda); 
//     }
//     if (p.p_diffperp==1)    {
//         dg::blas2::gemv( lapperp, y[0], lambda);
//         dg::blas2::gemv( lapperp, lambda, omega); //hyper
//         Dperp[0] = -p.nu_perp*dg::blas2::dot(one, w3d, omega); //hyper 
//     }
    if (p.p_torlim == 1)  {
         dsNU_.set_boundaries( p.bc, 0, 0); 
    }
    if (p.p_diff ==0)    {
    //     (A) adjoint
//         dsNU_( y[0], omega); 
//         dsNU_.centeredAdj(omega,lambda);
        dsNU_.forward( y[0], omega); 
        dsNU_.forwardAdj(omega,lambda);
        dg::blas1::axpby( 0.5, lambda, 0.,chi,chi); 
// 
        dsNU_.backward( y[0], omega); 
        dsNU_.backwardAdj(omega,lambda);
        dg::blas1::axpby( 0.5, lambda, 1., chi,chi); 
        Dpar[0]= p.nu_parallel*dg::blas2::dot(y[0], w3d, chi);
    }  
    if (p.p_diff ==1)    {
        // (B) nonadjoint
        dsNU_( m_invB, lambda); //gradpar 1/B
        dg::blas1::pointwiseDivide(lambda,  m_invB, lambda); //-ds lnB
        dsNU_(y[0],omega); //ds T
        dg::blas1::pointwiseDot(omega, lambda, omega);            // -ds lnB ds T
        dsNU_.dss(y[0],lambda);                                          //ds^2 T 
        dg::blas1::axpby( 1., omega,  1.,lambda );    
        Dpar[0]= p.nu_parallel*dg::blas2::dot(one, w3d, lambda);  
    }
    if (p.p_diff ==2)    {
        // (C) oldnonadjoint
        dsNU_.dss(y[0],omega);                                          //ds^2 N 
        dsNU_(y[0],lambda);       
        dg::blas1::pointwiseDot(gradlnB, lambda, lambda);            // ds lnB ds N
        dg::blas1::axpby( 1., omega, -1., lambda);       
        Dpar[0]= p.nu_parallel*dg::blas2::dot(one, w3d, lambda);  
    }
    if (p.p_diff ==3)    {
        // (D) nonadjoint with direct method
        dsNU_.forward( y[0], omega); 
        dg::blas1::pointwiseDot(omega, m_invB, omega);
        dsNU_.backward(omega,lambda);
        dg::blas1::pointwiseDivide(lambda, m_invB, lambda);
        dg::blas1::axpby( 0.5, lambda, 0.,chi,chi); 

        dsNU_.backward( y[0], omega); 
        dg::blas1::pointwiseDot(omega, m_invB, omega);
        dsNU_.forward(omega,lambda);
        dg::blas1::pointwiseDivide(lambda, m_invB, lambda);
        dg::blas1::axpby( 0.5, lambda, 1., chi,chi); 
//         Dpar[0]= p.nu_parallel*dg::blas2::dot(one, w3d, chi); 
        Dpar[0]= p.nu_parallel*dg::blas2::dot(y[0], w3d, chi); 
    }
    //Compute rhs of energy theorem
    ediff_= Dpar[0]+Dperp[0];
}


//do not overwrite y
template<class G, class I, class Matrix, class container>
void Explicit<G,I,Matrix,container>::operator()(const std::vector<container>& y, std::vector<container>& yp)
{
    /* y[0] := T - 1 or T
    */
    dg::Timer t;
    t.tic();
    assert( y.size() == 1);
    assert( y.size() == yp.size());
    dg::blas1::scal(yp[0],0.0);
/*    if (p.p_diffperp==0) {
        dg::blas2::gemv( lapperp, y[0], omega); //lap is negative
        dg::blas1::axpby( -p.nu_perp, omega, 0., yp[0]);  
    }
    if (p.p_diffperp==1) {
        dg::blas2::gemv( lapperp, y[0], omega); //lap is negative
        dg::blas2::gemv( lapperp, omega, lambda); //hyper
        dg::blas1::axpby( -p.nu_perp, lambda, 0., yp[0]);  //hyper 
    }  */ 

//     if (p.p_torlim == 1)  {
//          dsNU_.set_boundaries( p.bc, 0, 0); 
//     }   
// //-----------------------parallel adv------------------------
//     if (p.p_adv ==0)  {
//         // (A) adjoint
//         //U=v_parallel gradlnB
//     //     dg::blas1::pointwiseDot(y[0], gradlnB, lambda);
//     //     dsNU_.centeredAdj(lambda,omega);    
//     //     dg::blas1::axpby( p.nu_parallel, omega, 1., yp[0]); 
// 
//         //U=1.  
//         //centered
//         dg::blas1::pointwiseDot(y[0],pupil,lambda);    //U*T  
//         dsNU_.centeredAdj(lambda,omega);    // dsT UT
// //         dsNU_.backwardAdj(lambda,omega);
// //         dsNU_.forwardAdj(lambda,omega);   
//         dg::blas1::axpby( -1.0, omega, 1., yp[0]); //dsT (UT)
// 
//         //corr(1): div(UB) 
// //         dg::blas1::pointwiseDivide(pupil,m_invB,omega); //= U B
// //         dsNU_.centeredAdj(omega,lambda);     //div UB
// //         dg::blas1::axpby( 1.0, lambda, 1., yp[0]); //+div UB
//         
//         //corr(2): div(B) 
// //         dg::blas1::pointwiseDivide(one,m_invB,omega); //= U B
// //         dsNU_.centeredAdj(omega,lambda);     //div UB
// //         dg::blas1::axpby( 1.0, lambda, 1., yp[0]); //+div UB
//         
//         //corr(3): UT/B divB 
//     //     dg::blas1::pointwiseDivide(one,m_invB,omega); //=  B
//     //     dsNU_.centeredAdj(omega,lambda);     //div B
//     //     dg::blas1::pointwiseDot(y[0],m_invB,omega); //T/B
//     //     dg::blas1::pointwiseDot(omega,pupil,omega); // U T/B
//     //     dg::blas1::pointwiseDot(omega,lambda,lambda); //  UT/B divB
//     //     dg::blas1::axpby( 1.0, lambda, 1., yp[0]); //+ UT/B div UB
// 
//         //corr(4): U  divB
// //         dg::blas1::pointwiseDivide(one,m_invB,omega); //=  B
// //         dsNU_.centeredAdj(omega,lambda);     //div B
// //         dg::blas1::pointwiseDot(pupil,lambda,lambda); //  U divB
// //         dg::blas1::axpby( 1.0, lambda, 1., yp[0]); //+ U div UB
//     }
//     if (p.p_adv ==1)    {
//         // (B) nonadjoint U=1
//         dg::blas1::pointwiseDot(y[0],pupil,lambda);    //UT
//         dsNU_(lambda,omega);    //Dz UT
//         dg::blas1::axpby( -1.0, omega, 1., yp[0]); //-  ds U T
//         dsNU_( m_invB, lambda); //gradpar 1/B
//         dg::blas1::pointwiseDivide(lambda,  m_invB, lambda); //-ds lnB  
//         dg::blas1::pointwiseDot(y[0],pupil,omega);    //=  U T  
//         dg::blas1::pointwiseDot(omega,  lambda, omega); //-U T ds lnB  
//         dg::blas1::axpby( -1.0, omega, 1., yp[0]); //UT dslnB
//     }
//     if (p.p_adv ==2)    {
//         // (C) oldnonadjoint U=1
//         dg::blas1::pointwiseDot(y[0],pupil,lambda);    // UT
//         dsNU_(lambda,omega);    //  ds U T
//         dg::blas1::axpby( -1.0, omega, 1., yp[0]); //-  ds U T
//         dg::blas1::pointwiseDot(lambda,  gradlnB, omega); //U T ds lnB  
//         dg::blas1::axpby( 1.0, omega, 1., yp[0]); //UT dslnB
//     }
// //-----------------------parallel dissi------------------------
    if (p.p_diff ==0)    {
// //         centered
// //         dsNU_( y[0], omega); 
// //         dsNU_.centeredAdj(omega,lambda);
// //         dg::blas1::axpby( p.nu_parallel, lambda, 1., yp[0]); 
// 
        //forward, backward (stegi) without jump
//         dsNU_.forward( y[0], omega); 
//         dsNU_.forwardAdj(omega,lambda);
//         dg::blas1::axpby( 0.5*p.nu_parallel, lambda, 1., yp[0]); 
// 
//         dsNU_.backward( y[0], omega); 
//         dsNU_.backwardAdj(omega,lambda);
//         dg::blas1::axpby( 0.5*p.nu_parallel, lambda, 1., yp[0]); 
//         //with jump
//        dsNU_.symv(y[0],lambda);
// //        dg::blas1::pointwiseDot(w3d,lambda,lambda);
//        dg::blas1::axpby( p.nu_parallel, lambda, 1., yp[0]); 
// 
//         dsNU_.backward( y[0], omega); 
//         dsNU_.backwardAdj(omega,lambda);
//         dg::blas1::axpby( 0.5*p.nu_parallel, lambda, 1., yp[0]); 
        //with jump
//        dsNU_.symv(y[0],lambda);
// //        dg::blas1::pointwiseDot(v3d,lambda,lambda);
//        dg::blas1::axpby( p.nu_parallel, lambda, 1., yp[0]); 


    }

    if (p.p_diff ==1)    {
        // (B) nonadjoint
        dsNU_( m_invB, lambda); //gradpar 1/B
        dg::blas1::pointwiseDivide(lambda,  m_invB, lambda); //-ds lnB
        dsNU_(y[0],omega); //ds T
        dg::blas1::pointwiseDot(omega, lambda, omega);            //- ds lnB ds T
        dg::blas1::axpby(p.nu_parallel, omega, 1., yp[0]);    
        dsNU_.dss(y[0],omega);                                          //ds^2 T 
        dg::blas1::axpby( p.nu_parallel, omega, 1., yp[0]);
    }
    if (p.p_diff ==2)    {
        // (C) oldnonadjoint
        dsNU_.dss(y[0],omega);                                          //ds^2 T 
        dg::blas1::axpby( p.nu_parallel, omega, 1., yp[0]);       
        dsNU_(y[0],lambda);       
        dg::blas1::pointwiseDot(gradlnB, lambda, omega);            // ds lnB ds T
        dg::blas1::axpby(-p.nu_parallel, omega, 1., yp[0]);    
    }
    if (p.p_diff ==3)    {
        // (D) nonadjoint with direct method
        dsNU_.forward( y[0], omega); 
        dg::blas1::pointwiseDot(omega, m_invB, omega);
        dsDIR_.backward(omega,lambda);
        dg::blas1::pointwiseDivide(lambda, m_invB, lambda);
        dg::blas1::axpby( 0.5*p.nu_parallel, lambda, 1., yp[0]); 

        dsNU_.backward( y[0], omega); 
        dg::blas1::pointwiseDot(omega, m_invB, omega);
        dsDIR_.forward(omega,lambda);
        dg::blas1::pointwiseDivide(lambda, m_invB, lambda);

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

} //namespace heat

