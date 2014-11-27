#pragma once

#include "dg/algorithm.h"
#include "dg/poisson.h"
#include "feltor/parameters.h"
// #include "geometry_circ.h"
#include "solovev/geometry.h"
#include "solovev/init.h"

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
        temp( dg::evaluate(dg::zero, g)), chi(temp), omega(chi),
        dampprof_( dg::evaluate( solovev::GaussianProfDamping( gp), g)),
        dampgauss_( dg::evaluate( solovev::GaussianDamping( gp), g)),
        LaplacianM_perp ( g,g.bcx(),g.bcy(), dg::normed, dg::centered)
    {
    }
    void operator()( std::vector<container>& x, std::vector<container>& y)
    {
        /* x[0] := T - 1
        */
        dg::blas1::axpby( 0., x, 0, y);
        //not linear any more (cannot be written as y = Ax)
        dg::blas2::gemv( LaplacianM_perp, x[0], temp);
        dg::blas2::gemv( LaplacianM_perp, temp, y[0]);
        dg::blas1::scal( y[0], -p.nu_perp);  //  nu_perp lapl_RZ (lapl_RZ N) 

        //damping
        dg::blas1::pointwiseDot( dampgauss_, y[0], y[0]);

    }
    dg::Elliptic<Matrix, container, Preconditioner>& laplacianM() {return LaplacianM_perp;}
    const Preconditioner& weights(){return LaplacianM_perp.weights();}
    const Preconditioner& precond(){return LaplacianM_perp.precond();}
    const container& damping(){return dampprof_;}
  private:
    const eule::Parameters p;
    const solovev::GeomParameters gp;
    container temp, chi, omega;
    const container dampprof_;
    const container dampgauss_;
    
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

    template<class Grid3d>
    Feltor( const Grid3d& g, eule::Parameters p,solovev::GeomParameters gp);

    dg::DZ<Matrix, container> dz(){return dzDIR_;}

    void operator()( std::vector<container>& y, std::vector<container>& yp);

    double mass( ) {return mass_;}
    double mass_diffusion( ) {return diff_;}
    double energy( ) {return energy_;}
    std::vector<double> energy_vector( ) {return evec;}
    double energy_diffusion( ){ return ediff_;}

    void energies( std::vector<container>& y);

  private:


    container chi, omega, lambda; //!!Attention: chi and omega are helper variables and may be changed at any time and by any method!!

    const container binv, curvR, curvZ, gradlnB;
    const container source, damping, one;
    const Preconditioner w3d, v3d;

    //matrices and solvers
    dg::DZ<Matrix, container> dzDIR_;
    dg::DZ<Matrix, container> dzNU_;
    dg::Poisson< Matrix, container> poisson; 

    dg::Elliptic< Matrix, container, Preconditioner > lapperp; 


    const eule::Parameters p;
    const solovev::GeomParameters gp;

    double mass_, energy_, diff_, ediff_;
    std::vector<double> evec;

};

template<class Matrix, class container, class P>
template<class Grid>
Feltor<Matrix, container, P>::Feltor( const Grid& g, eule::Parameters p, solovev::GeomParameters gp): 
    chi( dg::evaluate( dg::one, g)), omega(chi),  lambda(chi), 
    binv( dg::evaluate(solovev::Field(gp) , g) ),
    curvR( dg::evaluate( solovev::CurvatureR(gp), g)),
    curvZ( dg::evaluate(solovev::CurvatureZ(gp), g)),
    gradlnB( dg::evaluate(solovev::GradLnB(gp) , g)),
    source( dg::evaluate(solovev::TanhSource(p, gp), g)),
    damping( dg::evaluate( solovev::GaussianDamping(gp ), g)), 
    one( dg::evaluate( dg::one, g)),    
    w3d( dg::create::weights(g)), v3d( dg::create::inv_weights(g)),      
    dzDIR_(solovev::Field(gp), g, 2.*M_PI/(double)p.Nz, gp.rk4eps,solovev::PsiLimiter(gp), dg::DIR),
    dzNU_(solovev::Field(gp), g, 2.*M_PI/(double)p.Nz, gp.rk4eps,solovev::PsiLimiter(gp), g.bcx()),
    poisson(g, g.bcx(), g.bcy(), dg::DIR, dg::DIR), //first N/U then phi BCC
    lapperp ( g,g.bcx(), g.bcy(),     dg::normed,         dg::centered),
    p(p),
    gp(gp),
    evec(1)
{ }



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
    dg::blas2::gemv( lapperp, y[0], lambda);
    dg::blas2::gemv( lapperp, lambda, omega);//nabla_RZ^4 N_e
    Dperp[0] = -p.nu_perp*dg::blas2::dot(one, w3d, omega);  
    //adjoint operator
    dzNU_( y[0], omega); 
    dzNU_.centeredT(omega,lambda);
    Dpar[0]= p.nu_parallel*dg::blas2::dot(one, w3d, lambda);  
    //adjoint but using old dz
/*    dzNU_( binv, lambda); //gradpar 1/B
    dg::blas1::pointwiseDivide(lambda,  binv, lambda); //dz lnB
    dzNU_(y[0],omega); //dz T
    dg::blas1::pointwiseDot(omega, lambda, omega);            // dz lnB dz N
    dzNU_.dzz(y[0],lambda);                                          //dz^2 T 
    dg::blas1::axpby( 1., omega,  1.,lambda );    
    Dpar[0]= p.nu_parallel*dg::blas2::dot(one, w3d, lambda); */ 

    //old
//     dzNU_.dzz(y[0],omega);                                          //dz^2 N 
//     dzNU_(y[0],lambda);       
//     dg::blas1::pointwiseDot(gradlnB, lambda, lambda);            // dz lnB dz N
//     dg::blas1::axpby( 1., omega, -1., lambda);       
//     Dpar[0]= p.nu_parallel*dg::blas2::dot(one, w3d, lambda);  
   

    //Compute rhs of energy theorem
    ediff_= Dpar[0]+Dperp[0];

}



template<class Matrix, class container, class P>
void Feltor<Matrix, container, P>::operator()( std::vector<container>& y, std::vector<container>& yp)
{
    /* y[0] := N_e - 1
    */
    dg::Timer t;
    t.tic();
    assert( y.size() == 1);
    assert( y.size() == yp.size());
    //compute phi via polarisation

    //parallel dynamics
    //adjoint operator
    dzNU_( y[0], omega); 
    dzNU_.centeredT(omega,lambda);
    dg::blas1::axpby( p.nu_parallel, lambda, 1., yp[0]); 

    //adjoint but using old dz
//     dzNU_( binv, lambda); //gradpar 1/B
//     dg::blas1::pointwiseDivide(lambda,  binv, lambda); //dz lnB
//     dzNU_(y[0],omega); //dz T
//     dg::blas1::pointwiseDot(omega, lambda, omega);            // dz lnB dz N
//     dg::blas1::axpby(p.nu_parallel, omega, 1., yp[0]);    
//     dzNU_.dzz(y[0],omega);                                          //dz^2 T 
//     dg::blas1::axpby( p.nu_parallel, omega, 1., yp[0]);
    
    //old laplace
//     dzNU_.dzz(y[0],omega);                                          //dz^2 T 
//     dg::blas1::axpby( p.nu_parallel, omega, 1., yp[0]);       
//     //gradlnBcorrection
//     dzNU_(y[0],lambda);       
//     dg::blas1::pointwiseDot(gradlnB, lambda, omega);            // dz lnB dz T
//     dg::blas1::axpby(-p.nu_parallel, omega, 1., yp[0]);    
    
    //damping 
    dg::blas1::pointwiseDot( damping, yp[0], yp[0]);

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

