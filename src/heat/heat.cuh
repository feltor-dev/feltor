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

namespace eule
{
///@addtogroup solver
///@{
/**
 * @brief Diffusive terms for Feltor solver
 *
 * @tparam Matrix The Matrix class
 * @tparam container The Vector class 
 * @tparam container The container class
 */
template< class Geometry, class DS, class Matrix, class container>
struct Rolkar
{
    Rolkar( const Geometry& g, eule::Parameters p, dg::geo::solovev::GeomParameters gp):
        p(p),
        gp(gp),
        dsNU_( typename DS::FieldAligned(
                dg::geo::Field<dg::geo::solovev::MagneticField>(
                    dg::geo::solovev::MagneticField(gp), gp.R_0), 
                g, gp.rk4eps, 
                dg::geo::PsiLimiter<dg::geo::solovev::Psip>(
                    dg::geo::solovev::Psip(gp), gp.psipmaxlim
                    ), 
                g.bcx()
                ), 
            dg::geo::Field<dg::geo::solovev::MagneticField>(
              dg::geo::solovev::MagneticField(gp), gp.R_0
              ), dg::normed, dg::forward ),
        elliptic( g, dg::normed, dg::forward)
    {
        using namespace dg::geo::solovev;
        MagneticField c(gp);
        container bfield = dg::evaluate( dg::geo::FieldR<MagneticField>(c, gp.R_0),g);
        elliptic.set_x( bfield);
        bfield = dg::evaluate( dg::geo::FieldZ<MagneticField>(c, gp.R_0),g);
        elliptic.set_y( bfield);
        bfield = dg::evaluate( dg::geo::FieldP<MagneticField>(c, gp.R_0),g);
        elliptic.set_z( bfield);

        dg::blas1::transfer( dg::pullback( dg::geo::GaussianProfDamping<Psip>(Psip(gp), gp.psipmax, gp.alpha), g), dampprof_);
    }
    void operator()( std::vector<container>& x, std::vector<container>& y)
    {
        dg::blas1::axpby( 0., x, 0, y);
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
    const container& precond(){return elliptic.precond();}
  private:
    const eule::Parameters p;
    const dg::geo::solovev::GeomParameters gp;
    container dampprof_;
    DS dsNU_;
    dg::GeneralEllipticSym<Geometry, Matrix, container> elliptic;

};

template< class DS, class Matrix, class container >
struct Feltor
{
    //typedef std::vector<container> Vector;
    typedef typename dg::VectorTraits<container>::value_type value_type;
    //typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    //typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    //typedef dg::DMatrix Matrix; //fastest device Matrix (does this conflict with 

    template<class Grid3d>
    Feltor( const Grid3d& g, eule::Parameters p,dg::geo::solovev::GeomParameters gp);

    const DS ds(){return dsNU_;}

    void operator()( std::vector<container>& y, std::vector<container>& yp);

    double mass( ) {return mass_;}
    double mass_diffusion( ) {return diff_;}
    double energy( ) {return energy_;}
    std::vector<double> energy_vector( ) {return evec;}
    double energy_diffusion( ){ return ediff_;}
    void energies( std::vector<container>& y);

  private:


    container chi, omega, lambda,tmo; //!!Attention: chi and omega are helper variables and may be changed at any time and by any method!!
    container binv, gradlnB;
//     ,pupil;
    const container  one;
    const container w3d, v3d;

    //matrices and solvers
    DS dsDIR_, dsNU_;

//     dg::Elliptic< Matrix, container > lapperp; 
//     dg::GeneralEllipticSym<Matrix, container> elliptic;

    const eule::Parameters p;
    const dg::geo::solovev::GeomParameters gp;
    std::vector<double> evec;

    double mass_, energy_, diff_, ediff_;

};

template<class DS, class Matrix, class container>
template<class Grid>
Feltor<DS, Matrix, container>::Feltor( const Grid& g, eule::Parameters p, dg::geo::solovev::GeomParameters gp): 
    chi( dg::evaluate( dg::one, g)), omega(chi),  lambda(chi), tmo(chi),
    one( dg::evaluate( dg::one, g)),    
    w3d( dg::create::volume(g)), v3d( dg::create::inv_volume(g)),      
    dsDIR_( typename DS::FieldAligned( 
                dg::geo::Field<dg::geo::solovev::MagneticField>(
                    dg::geo::solovev::MagneticField(gp), gp.R_0
                    ), 
                g, gp.rk4eps, 
                dg::geo::PsiLimiter<dg::geo::solovev::Psip>(
                    dg::geo::solovev::Psip(gp), gp.psipmaxlim
                    ), 
                dg::DIR
                ), 
            dg::geo::Field<dg::geo::solovev::MagneticField>(
                dg::geo::solovev::MagneticField(gp), gp.R_0
                ), 
            dg::normed, dg::centered ),
    dsNU_( typename DS::FieldAligned(
                dg::geo::Field<dg::geo::solovev::MagneticField>(
                    dg::geo::solovev::MagneticField(gp), gp.R_0), 
                g, gp.rk4eps, 
                dg::geo::PsiLimiter<dg::geo::solovev::Psip>(
                    dg::geo::solovev::Psip(gp), gp.psipmaxlim
                    ), 
                g.bcx()
                ), 
          dg::geo::Field<dg::geo::solovev::MagneticField>(
              dg::geo::solovev::MagneticField(gp), gp.R_0
              ), 
          dg::normed, dg::centered ),
//     lapperp ( g,g.bcx(), g.bcy(),     dg::normed,  dg::centered),
//         elliptic( g, dg::normed, dg::forward),
    p(p),
    gp(gp),
    evec(1)
{
//         container bfield = dg::evaluate( solovev::bR( gp.R_0, gp.I_0),g);
//         elliptic.set_x( bfield);
//         bfield = dg::evaluate( solovev::bZ( gp.R_0, gp.I_0),g);
//         elliptic.set_y( bfield);
//         bfield = dg::evaluate( solovev::bPhi( gp.R_0, gp.I_0),g);
//         elliptic.set_z( bfield);
    //////////////////////////////init fields /////////////////////
    using namespace dg::geo::solovev;
    MagneticField mf(gp);
    dg::blas1::transfer(  dg::pullback(dg::geo::Field<MagneticField>(mf, gp.R_0),                     g), binv);
    dg::blas1::transfer(  dg::pullback(dg::geo::GradLnB<MagneticField>(mf, gp.R_0),                   g), gradlnB);
}



template<class DS, class M, class V>
void Feltor<DS, M, V>::energies( std::vector<V>& y)
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
//         dsNU_.centeredT(omega,lambda);
        dsNU_.forward( y[0], omega); 
        dsNU_.forwardT(omega,lambda);
        dg::blas1::axpby( 0.5, lambda, 0.,chi,chi); 
// 
        dsNU_.backward( y[0], omega); 
        dsNU_.backwardT(omega,lambda);
        dg::blas1::axpby( 0.5, lambda, 1., chi,chi); 
        Dpar[0]= p.nu_parallel*dg::blas2::dot(y[0], w3d, chi);
    }  
    if (p.p_diff ==1)    {
        // (B) nonadjoint
        dsNU_( binv, lambda); //gradpar 1/B
        dg::blas1::pointwiseDivide(lambda,  binv, lambda); //-ds lnB
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
        dsNU_.forwardTD(omega,lambda);
        dg::blas1::axpby( 0.5, lambda, 0.,chi,chi); 

        dsNU_.backward( y[0], omega); 
        dsNU_.backwardTD(omega,lambda);
        dg::blas1::axpby( 0.5, lambda, 1., chi,chi); 
//         Dpar[0]= p.nu_parallel*dg::blas2::dot(one, w3d, chi); 
        Dpar[0]= p.nu_parallel*dg::blas2::dot(y[0], w3d, chi); 
    }
    //Compute rhs of energy theorem
    ediff_= Dpar[0]+Dperp[0];
}


//do not overwrite y
template<class DS, class Matrix, class container>
void Feltor<DS, Matrix, container>::operator()( std::vector<container>& y, std::vector<container>& yp)
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
//     //     dsNU_.centeredT(lambda,omega);    
//     //     dg::blas1::axpby( p.nu_parallel, omega, 1., yp[0]); 
// 
//         //U=1.  
//         //centered
//         dg::blas1::pointwiseDot(y[0],pupil,lambda);    //U*T  
//         dsNU_.centeredT(lambda,omega);    // dsT UT
// //         dsNU_.backwardT(lambda,omega);
// //         dsNU_.forwardT(lambda,omega);   
//         dg::blas1::axpby( -1.0, omega, 1., yp[0]); //dsT (UT)
// 
//         //corr(1): div(UB) 
// //         dg::blas1::pointwiseDivide(pupil,binv,omega); //= U B
// //         dsNU_.centeredT(omega,lambda);     //div UB
// //         dg::blas1::axpby( 1.0, lambda, 1., yp[0]); //+div UB
//         
//         //corr(2): div(B) 
// //         dg::blas1::pointwiseDivide(one,binv,omega); //= U B
// //         dsNU_.centeredT(omega,lambda);     //div UB
// //         dg::blas1::axpby( 1.0, lambda, 1., yp[0]); //+div UB
//         
//         //corr(3): UT/B divB 
//     //     dg::blas1::pointwiseDivide(one,binv,omega); //=  B
//     //     dsNU_.centeredT(omega,lambda);     //div B
//     //     dg::blas1::pointwiseDot(y[0],binv,omega); //T/B
//     //     dg::blas1::pointwiseDot(omega,pupil,omega); // U T/B
//     //     dg::blas1::pointwiseDot(omega,lambda,lambda); //  UT/B divB
//     //     dg::blas1::axpby( 1.0, lambda, 1., yp[0]); //+ UT/B div UB
// 
//         //corr(4): U  divB
// //         dg::blas1::pointwiseDivide(one,binv,omega); //=  B
// //         dsNU_.centeredT(omega,lambda);     //div B
// //         dg::blas1::pointwiseDot(pupil,lambda,lambda); //  U divB
// //         dg::blas1::axpby( 1.0, lambda, 1., yp[0]); //+ U div UB
//     }
//     if (p.p_adv ==1)    {
//         // (B) nonadjoint U=1
//         dg::blas1::pointwiseDot(y[0],pupil,lambda);    //UT
//         dsNU_(lambda,omega);    //Dz UT
//         dg::blas1::axpby( -1.0, omega, 1., yp[0]); //-  ds U T
//         dsNU_( binv, lambda); //gradpar 1/B
//         dg::blas1::pointwiseDivide(lambda,  binv, lambda); //-ds lnB  
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
// //         dsNU_.centeredT(omega,lambda);
// //         dg::blas1::axpby( p.nu_parallel, lambda, 1., yp[0]); 
// 
        //forward, backward (stegi) without jump
//         dsNU_.forward( y[0], omega); 
//         dsNU_.forwardT(omega,lambda);
//         dg::blas1::axpby( 0.5*p.nu_parallel, lambda, 1., yp[0]); 
// 
//         dsNU_.backward( y[0], omega); 
//         dsNU_.backwardT(omega,lambda);
//         dg::blas1::axpby( 0.5*p.nu_parallel, lambda, 1., yp[0]); 
//         //with jump
//        dsNU_.symv(y[0],lambda);
// //        dg::blas1::pointwiseDot(w3d,lambda,lambda);
//        dg::blas1::axpby( p.nu_parallel, lambda, 1., yp[0]); 
// 
//         dsNU_.backward( y[0], omega); 
//         dsNU_.backwardT(omega,lambda);
//         dg::blas1::axpby( 0.5*p.nu_parallel, lambda, 1., yp[0]); 
        //with jump
//        dsNU_.symv(y[0],lambda);
// //        dg::blas1::pointwiseDot(v3d,lambda,lambda);
//        dg::blas1::axpby( p.nu_parallel, lambda, 1., yp[0]); 


    }

    if (p.p_diff ==1)    {
        // (B) nonadjoint
        dsNU_( binv, lambda); //gradpar 1/B
        dg::blas1::pointwiseDivide(lambda,  binv, lambda); //-ds lnB
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
        dsDIR_.forwardTD(omega,lambda);
        dg::blas1::axpby( 0.5*p.nu_parallel, lambda, 1., yp[0]); 

        dsNU_.backward( y[0], omega); 
        dsDIR_.backwardTD(omega,lambda);

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

