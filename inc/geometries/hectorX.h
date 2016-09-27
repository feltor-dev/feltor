#pragma once

#include "hector.h"
#include "interpolationX.cuh"
#include "orthogonalX.h"



namespace dg
{

/**
 * @brief The High PrEcision Conformal grid generaTOR
 *
 * @tparam IMatrix The interpolation matrix type
 * @tparam Matrix  The matrix type in the elliptic equation
 * @tparam container The container type for internal computations (must be compatible to a thrust::host_vector<double> in the blas1::transfer function)
 */
template <class IMatrix = dg::IHMatrix, class Matrix = dg::Composite<dg::HMatrix>, class container = dg::HVec>
struct HectorX
{
    typedef dg::ConformalTag metric_category; //!This typedef is for the construction of a dg::conformal::Grid

    /**
     * @brief Construct from functors
     *
     * @tparam Psi A binary functor
     * @tparam PsiX The first derivative in x
     * @tparam PsiY The first derivative in y
     * @tparam LaplacePsi The Laplacian function 
     * @param psi The function 
     * @param psiX The first derivative in x 
     * @param psiY The first derivative in y
     * @param laplacePsi The Laplacian 
     * @param psi0 first boundary 
     * @param psi1 second boundary
     * @param X0 a point in the inside of the ring bounded by psi0
     * @param Y0 a point in the inside of the ring bounded by psi0
     * @param n number of polynomials used for the orthogonal grid
     * @param Nx initial number of points in zeta
     * @param Ny initial number of points in eta
     * @param eps_u the accuracy of u
     */
    template< class Psi, class PsiX, class PsiY, class LaplacePsi>
    HectorX( Psi psi, PsiX psiX, PsiY psiY, LaplacePsi laplacePsi, double psi0, double psi1, double XX, double YX, double X0, double Y0, double fx, double fy, unsigned n = 13, unsigned Nx = 2, unsigned Ny = 20, double eps_u = 1e-10) : 
        g2d_(dg::SeparatrixOrthogonal<Psi,PsiX,PsiY,LaplacePsi>(psi, psiX, psiY, laplacePsi, psi0, psi1, X0, Y0,0), psi_0, fx, fy, n, Nx, Ny, dg::DIR)
    {
        //first construct u_
        container u = construct_grid_and_u( psi, psiX, psiY, laplacePsi, psi0, psi1, XX, YX, X0, Y0, fx, fy, n, Nx, Ny, eps_u );
        //now compute u_zeta and u_eta 
        //Matrix dzeta = dg::create::dx( g2d_, dg::DIR);
        //Matrix deta = dg::create::dy( g2d_, dg::PER);
        //container u_zeta(u), u_eta(u);
        //dg::blas2::symv( dzeta, u, u_zeta);
        //dg::blas1::plus( u_zeta, 1.);
        //dg::blas2::symv( deta, u, u_eta);

        ////we actually don't need u but it makes a good testcase 
        //container zeta;
        //dg::blas1::transfer(dg::evaluate( dg::cooX2d, g2d_), zeta);
        //dg::blas1::axpby( +1., zeta, 1.,  u); //u = \tilde u + \zeta

        ////now compute ZetaU and EtaU
        //container a2;
        //dg::blas1::transfer( g2d_.g_yy(), a2);
        //dg::blas1::pointwiseDivide( a2, g2d_.g_xx(), a2); // a^2=g^ee/g^zz
        //container zetaU=u_zeta, etaU=u_eta;
        //dg::blas1::pointwiseDot( zetaU, zetaU, zetaU); //u_z*u_z
        //dg::blas1::pointwiseDot(  etaU,  etaU,  etaU); //u_e*u_e
        //dg::blas1::pointwiseDot( 1., etaU, a2, 1., zetaU); //u_z*u_z+a^2u_e*u_e
        //container den( zetaU); //denominator
        //dg::blas1::pointwiseDivide( u_zeta, den, zetaU); //u_z / denominator
        //dg::blas1::pointwiseDivide(  u_eta, den,  etaU); 
        //dg::blas1::pointwiseDot( a2, etaU, etaU); //a^2*u_e / denominator
        ////now compute etaV and its inverse
        //container etaV(etaU), ones(etaU.size(), 1.);
        //dg::blas1::pointwiseDivide( ones, u_zeta, etaV);
        //thrust::host_vector<double> etaVinv_h;
        //dg::blas1::transfer( u_zeta, etaVinv_h);

        ////construct c0 and scale all vector components with it
        //c0_ = detail::construct_c0( etaVinv_h, g2d_);
        //dg::blas1::scal(  etaV, 1./c0_);
        //dg::blas1::scal( zetaU, 1./c0_);
        //dg::blas1::scal(  etaU, 1./c0_);
        //dg::blas1::scal( u_zeta, c0_);
        //dg::blas1::scal(  u_eta, c0_);
        //dg::blas1::scal(  u, c0_);
        ////transfer to host
        //dg::blas1::transfer( u, u_);
        //detail::transform( u_zeta, u_eta, ux_, uy_, g2d_);
        //dg::blas1::transfer( etaV, etaV_);
        //dg::blas1::transfer( etaU, etaU_);
        //dg::blas1::transfer( zetaU, zetaU_);
        ////std::cout << "c0 is "<<c0_<<"\n";
    }
    /**
     * @brief The length of the u domain
     *
     * Call before discreizing the u domain
     * @return  
     */
    double lu() const {return c0_*g2d_.lx();}
    /**
     * @brief The length of the v domain
     *
     * Always returns 2pi
     * @return 2pi 
     */
    double lv() const {return 2.*M_PI;}
    /**
     * @brief Generate the points and the elements of the Jacobian
     *
     * @param u1d one-dimensional list of points inside the u-domain
     * @param v1d one-dimensional list of points inside the v-domain
     * @param x  = x(u,v)
     * @param y  = y(u,v)
     * @param ux = u_x(u,v)
     * @param uy = u_y(u,v)
     * @param vx = -u_y(u,v)
     * @param vy = u_x(u,v)
     * @note All the resulting vectors are write-only and get properly resized
     */
    void operator()( const thrust::host_vector<double>& u1d, 
                     const thrust::host_vector<double>& v1d, 
                     thrust::host_vector<double>& x, 
                     thrust::host_vector<double>& y, 
                     thrust::host_vector<double>& ux, 
                     thrust::host_vector<double>& uy, 
                     thrust::host_vector<double>& vx, 
                     thrust::host_vector<double>& vy) const
    {
        thrust::host_vector<double> eta_init, zeta, eta; 
        detail::compute_zev( etaV_, v1d, eta_init, g2d_);
        thrust::host_vector<double> zeta_init( eta_init.size(), 0.); 
        detail::construct_grid( zetaU_, etaU_, u1d, zeta_init, eta_init, zeta, eta, g2d_);
        //the box is periodic in eta and the y=0 line needs not to coincide with the eta=0 line
        for( unsigned i=0; i<eta.size(); i++)
            eta[i] = fmod(eta[i]+2.*M_PI, 2.*M_PI); 
        dg::IHMatrix Q = dg::create::interpolation( zeta, eta, g2d_);

        thrust::host_vector<double> u(u1d.size()*v1d.size());
        x = y = ux = uy = vx = vy = u;  //resize 
        dg::blas2::symv( Q, g2d_.r(), x);
        dg::blas2::symv( Q, g2d_.z(), y);
        dg::blas2::symv( Q, ux_, ux);
        dg::blas2::symv( Q, uy_, uy);
        dg::blas1::transfer( ux, vy);
        dg::blas1::axpby( -1., uy, 0., vx);


    }

    /**
     * @brief Return the internally used orthogonal grid
     *
     * @return  orthogonal zeta, eta grid
     */
    const dg::orthogonal::GridX2d<container>& orthogonal_grid() const {return g2d_;}
    private:
    template< class Psi, class PsiX, class PsiY, class LaplacePsi>
    container construct_grid_and_u( Psi psi, PsiX psiX, PsiY psiY, LaplacePsi laplacePsi, double psi0, double psi1, double XX, double YX, double X0, double Y0, double fx, double fy, unsigned n, unsigned Nx, unsigned Ny, double eps_u ) 
    {
        //first find u( \zeta, \eta)
        double eps = 1e10, eps_old = 2e10;
        dg::SeparatrixOrthogonal<Psi,PsiX,PsiY,LaplacePsi> generator(psi, psiX, psiY, laplacePsi, psi0, XX,YX, X0, Y0,1);
        dg::orthogonal::GridX2d<container> g2d_old(generator, psi_0, fx, fy, n, Nx, Ny, dg::DIR, dg::NEU);
        dg::Elliptic<dg::orthogonal::GridX2d<container>, Matrix, container> ellipticD_old( g2d_old, dg::DIR, dg::NEU, dg::not_normed, dg::centered);

        container u_old = dg::evaluate( dg::zero, g2d_old), u(u_old);
        container lapu = g2d_old.lapx();
        dg::Invert<container > invert_old( u_old, n*n*Nx*Ny, eps_u);
        unsigned number = invert_old( ellipticD_old, u_old, lapu);
        while( (eps < eps_old||eps > 1e-7) && eps > eps_u)
        {
            eps = eps_old;
            Nx*=2, Ny*=2;
            dg::orthogonal::GridX2d<container> g2d(generator, psi_0, fx, fy, n, Nx, Ny, dg::DIR, dg::NEU);
            dg::Elliptic<dg::orthogonal::GridX2d<container>, Matrix, container> ellipticD( g2d, dg::DIR, dg::NEU, dg::not_normed, dg::centered);
            lapu = g2d.lapx();
            const container vol2d = dg::create::weights( g2d);
            const IMatrix Q = dg::create::interpolation( g2d, g2d_old);
            u = dg::evaluate( dg::zero, g2d);
            container u_diff( u);
            dg::blas2::gemv( Q, u_old, u_diff);

            dg::Invert<container > invert( u_diff, n*n*Nx*Ny, 0.1*eps_u);
            number = invert( ellipticD, u, lapu);
            dg::blas1::axpby( 1. ,u, -1., u_diff);
            eps = sqrt( dg::blas2::dot( u_diff, vol2d, u_diff) / dg::blas2::dot( u, vol2d, u) );
            std::cout << "Nx "<<Nx<<" Ny "<<Ny<<" error "<<eps<<"\n";
            g2d_old = g2d;
            u_old = u;
            g2d_ = g2d;
            number++;//get rid of warning
        }
        return u;
    }

    double c0_;
    thrust::host_vector<double> u_, ux_, uy_;
    thrust::host_vector<double> etaV_, zetaU_, etaU_;
    dg::orthogonal::GridX2d<container> g2d_;

};

}//namespace dg
