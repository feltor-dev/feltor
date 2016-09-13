#pragma once

#include "dg/backend/grid.h"
#include "dg/elliptic.h"
#include "orthogonal.h"



namespace hector
{

//container must be compliant in blas1::transfer function
template <class IMatrix, class Matrix, class container>
struct Hector
{
    template< class Psi, class PsiX, class PsiY, class PsiXX, class PsiXY, class PsipYY, class LaplacePsiX, class LaplacePsiY>
    Hector( Psi psi, PsiX psiX, PsiY psiY, PsiXX psiXX, PsiXY psiXY, PsiYY psiYY, LaplacePsiX laplacePsiX, LaplacePsiY laplacePsiY, double psi0, double psi1, double X0, double Y0, unsigned n, unsigned Nx, unsigned Ny, double eps) : 
        g2d_(psi, psiX, psiY, psiXX, psiXY, psiYY, laplacePsiX, laplacePsiY, psi0, psi1, X0, Y0, n, Nx, Ny)
    {
        double eps = 1e10, eps_old = 2e10;
        orthogonal::RingGrid2d<dg::DVec> g2d_old(gp, psi_0, psi_1, n, Nx, Ny,dg::NEU);
        dg::Elliptic<orthogonal::RingGrid2d<container>, Matrix, container> ellipticD_old( g2d_old, dg::DIR, dg::PER, dg::not_normed, dg::centered);
        dg::Elliptic<orthogonal::RingGrid2d<container>, Matrix, container> ellipticN_old( g2d_old, dg::DIR_NEU, dg::PER, dg::not_normed, dg::centered);

        container u_old = dg::evaluate( dg::zero, g2d_old);
        container v_old = dg::evaluate( dg::zero, g2d_old);
        container lapu = g2d_old.lapy();
        container lapv = g2d_old.lapx();
        dg::Invert<container > invert_old( u_old, n*n*Nx*Ny, 1e-10);
        unsigned number = invert_old( ellipticD_old, u_old, lapu);
        unsigned number = invert_old( ellipticN_old, v_old, lapv);
        while( (eps < eps_old||eps > 1e-7) && eps > 1e-10)
        {
            eps = eps_old;
            Nx*=2, Ny*=2;
            orthogonal::RingGrid2d<container> g2d(gp, psi_0, psi_1, n, Nx, Ny,dg::NEU);
            dg::Elliptic<orthogonal::RingGrid2d<container>, Matrix, container> ellipticD( g2d, dg::DIR, dg::PER, dg::not_normed, dg::centered);
            dg::Elliptic<orthogonal::RingGrid2d<container>, Matrix, container> ellipticN( g2d, dg::DIR_NEU, dg::PER, dg::not_normed, dg::centered);
            lapu = g2d.lapx();
            lapv = g2d.lapy();
            const container vol2d = dg::create::weights( g2d);

            const IMatrix Q = dg::create::interpolation( g2d, g2d_old);
            container u = dg::evaluate( dg::zero, g2d), u_diff( u);
            container v = dg::evaluate( dg::zero, g2d), v_diff( v);
            dg::blas2::gemv( Q, u_old, u_diff);
            dg::blas2::gemv( Q, v_old, v_diff);

            dg::Invert<container > invert( u_diff, n*n*Nx*Ny, 1e-10);
            dg::Invert<container > invert( v_diff, n*n*Nx*Ny, 1e-10);
            number = invert( ellipticD, u, lapu);
            number = invert( ellipticN, u, lapv);
            dg::blas1::axpby( 1. ,u, -1., u_diff);
            dg::blas1::axpby( 1. ,v, -1., v_diff);
            eps = sqrt( dg::blas2::dot( u_diff, vol2d, u_diff) / dg::blas2::dot( u, vol2d, u) );
            eps = sqrt( dg::blas2::dot( v_diff, vol2d, v_diff) / dg::blas2::dot( v, vol2d, v) );
            //std::cout << "Nx "<<Nx<<" Ny "<<Ny<<" error "<<eps<<"\n";
            g2d_old = g2d;
            u_old = u;
            v_old = v;
            g2d_ = g2d;
        }
        Matrix dzeta = dg::create::dx( g2d_, dg::DIR);
        Matrix deta = dg::create::dy( g2d_, dg::DIR_NEU);
        container u_zeta(u), u_eta(u), v_zeta(v), v_zeta(v);
        container u_x(u), u_y(u), v_x(v), v_y(v);
        dg::blas2::symv( dzeta, u, u_zeta);
        dg::blas1::plus( u_zeta, 1.);
        dg::blas2::symv( deta, u, u_eta);
        dg::blas2::symv( dzeta, v, v_zeta);
        dg::blas2::symv( deta, v, v_eta);
        dg::blas1::plus( v_eta, 1.);
        dg::blas1::pointwiseDot( u_zeta, g2d.xr(), u_x); 
        dg::blas1::pointwiseDot( 1., u_eta, g2d.yr(), 1., u_x); 
        dg::blas1::pointwiseDot( u_zeta, g2d.xz(), u_y); 
        dg::blas1::pointwiseDot( 1., u_eta, g2d.yz(), 1., u_y); 
        dg::blas1::pointwiseDot( v_zeta, g2d.xr(), v_x); 
        dg::blas1::pointwiseDot( 1., u_eta, g2d.yr(), 1., v_x); 
        dg::blas1::pointwiseDot( v_zeta, g2d.xz(), v_y); 
        dg::blas1::pointwiseDot( 1., v_eta, g2d.yz(), 1., v_y); 

        container xdiff(u), ydiff(u);
        dg::blas1::axpby( 1., u_x, -1., v_y, xdiff);
        dg::blas1::axpby( 1., u_y, +1., v_x, ydiff);
        dg::blas2::dot( xdiff, vol2d, xdiff);
        dg::blas2::dot( ydiff, vol2d, ydiff);
    }
    double lu() const {return g2d_.lx();}
    double lv() const {return 2.*M_PI;}
    void operator()( const thrust::host_vector<double>& u, 
                     const thrust::host_vector<double>& v, 
                     thrust::host_vector<double>& x, 
                     thrust::host_vector<double>& y, 
                     thrust::host_vector<double>& ux, 
                     thrust::host_vector<double>& uy, 
                     thrust::host_vector<double>& vx, 
                     thrust::host_vector<double>& vy);
    private
    orthogonal::RingGrid2d<container> g2d_;

};

}//namespace hector
