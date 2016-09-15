#pragma once

#include <vector>
#include "dg/backend/grid.h"
#include "dg/backend/interpolation.cuh"
#include "dg/elliptic.h"
#include "dg/cg.h"
#include "orthogonal.h"



namespace hector
{
namespace detail
{

//interpolate the two components of a vector field
struct Interpolate
{
    Interpolate( const thrust::host_vector<double>& fZeta, 
                 const thrust::host_vector<double>& fEta, 
                 const dg::Grid2d<double>& g2d ): 
        iter0_( dg::create::forward_transform( fZeta, g2d) ), 
        iter1_( dg::create::forward_transform(  fEta, g2d) ), 
        g_(g2d), zeta1_(g2d.x1()), eta1_(g2d.y1()){}
    void operator()(const thrust::host_vector<double>& zeta, thrust::host_vector<double>& fZeta)
    {
        fZeta[0] = interpolate( fmod( zeta[0]+zeta1_, zeta1_), fmod( zeta[1]+eta1_, eta1_), iter0_, g_);
        fZeta[1] = interpolate( fmod( zeta[0]+zeta1_, zeta1_), fmod( zeta[1]+eta1_, eta1_), iter1_, g_);
        //fZeta[0] = interpolate(  zeta[0], zeta[1], iter0_, g_);
        //fZeta[1] = interpolate(  zeta[0], zeta[1], iter1_, g_);
    }
    void operator()(const std::vector<thrust::host_vector<double> >& zeta, std::vector< thrust::host_vector<double> >& fZeta)
    {
        for( unsigned i=0; i<zeta[0].size(); i++)
        {
            fZeta[0][i] = interpolate( fmod( zeta[0][i]+zeta1_, zeta1_), fmod( zeta[1][i]+eta1_, eta1_), iter0_, g_);
            fZeta[1][i] = interpolate( fmod( zeta[0][i]+zeta1_, zeta1_), fmod( zeta[1][i]+eta1_, eta1_), iter1_, g_);
        }
    }
    private:
    thrust::host_vector<double> iter0_;
    thrust::host_vector<double> iter1_;
    dg::Grid2d<double> g_;
    double zeta1_, eta1_;
};

//compute c_0 
double construct_c0( const thrust::host_vector<double>& etaVinv, const dg::Grid2d<double>& g2d) 
{
    //this is a normal integration:
    thrust::host_vector<double> etaVinvL( dg::create::forward_transform(  etaVinv, g2d) );
    dg::Grid1d<double> g1d( 0., 2.*M_PI, g2d.n(), g2d.Ny());
    dg::HVec eta = dg::evaluate(dg::coo1, g1d);
    dg::HVec w1d = dg::create::weights( g1d);
    dg::HVec int_etaVinv(eta);
    for( unsigned i=0; i<eta.size(); i++) 
        int_etaVinv[i] = interpolate( 0., eta[i], etaVinvL, g2d);
    double c0 = 2.*M_PI/dg::blas1::dot( w1d, int_etaVinv );
    return c0;

    //for some reason the following is a bad idea (gives a slightly wrong result):
    //Interpolate inter( thrust::host_vector<double>( etaVinv.size(), 0), etaVinv, g2d);
    //thrust::host_vector<double> begin( 2, 0), end(begin), end_old(begin);
    //begin[0] = 0, begin[1] = 0;
    //double eps = 1e10, eps_old = 2e10;
    //unsigned N = 5;
    //while( (eps < eps_old || eps > 1e-7)&& eps > 1e-12)
    //{
    //    eps_old = eps, end_old = end;
    //    N*=2; dg::stepperRK4( inter, begin, end, 0., 2*M_PI, N);
    //    eps = fabs( end[1]-end_old[1]);
    //    std::cout << "\t error eps "<<eps<<" with "<<N<<" steps: " << 2*M_PI/end[1]<<"\n";
    //    std::cout << "\t error c0  "<<fabs(c0-2.*M_PI/end[1])<<" with "<<N<<" steps: " << 2*M_PI/end[1]<<"\n";
    //}
    ////std::cout <<end_old[2] << " "<<end[2] << "error in y is "<<y_eps<<"\n";
    //double f_psi = 2.*M_PI/end_old[1];

    //return f_psi;
}


//compute the vector of zeta and eta - values that form first v surface
void compute_zev( 
        const thrust::host_vector<double>& etaV,
        const thrust::host_vector<double>& v_vec,
        thrust::host_vector<double>& eta, 
        const dg::Grid2d<double>& g2d
        ) 
{
    Interpolate iter( thrust::host_vector<double>( etaV.size(), 0), etaV, g2d);
    eta.resize( v_vec.size());
    thrust::host_vector<double> eta_old(v_vec.size(), 0), eta_diff( eta_old);
    thrust::host_vector<double> begin( 2, 0), end(begin), temp(begin);
    begin[0] = 0., begin[1] = 0.;
    unsigned steps = 1;
    double eps = 1e10, eps_old=2e10;
    while( (eps < eps_old||eps > 1e-7) && eps > 1e-12)
    {
        //begin is left const
        eps_old = eps, eta_old = eta;
        dg::stepperRK17( iter, begin, end, 0, v_vec[0], steps);
        eta[0] = end[1];
        for( unsigned i=1; i<v_vec.size(); i++)
        {
            temp = end;
            dg::stepperRK17( iter, temp, end, v_vec[i-1], v_vec[i], steps);
            eta[i] = end[1];
        }
        temp = end;
        dg::stepperRK17( iter, temp, end, v_vec[v_vec.size()-1], 2.*M_PI, steps);
        dg::blas1::axpby( 1., eta, -1., eta_old, eta_diff);
        eps =  sqrt( dg::blas1::dot( eta_diff, eta_diff) / dg::blas1::dot( eta, eta));
        //std::cout << "rel. error is "<<eps<<" with "<<steps<<" steps\n";
        //std::cout << "abs. error is "<<( 2.*M_PI-end[1])<<"\n";
        steps*=2;
    }
}

void construct_grid( 
        const thrust::host_vector<double>& zetaU, //2d Zeta component
        const thrust::host_vector<double>& etaU,  //2d Eta component
        const thrust::host_vector<double>& u_vec,  //1d u values
        const thrust::host_vector<double>& zeta_init, //1d intial values
        const thrust::host_vector<double>& eta_init, //1d intial values
        thrust::host_vector<double>& zeta, 
        thrust::host_vector<double>& eta, 
        const dg::Grid2d<double>& g2d
    )
{
    Interpolate inter( zetaU, etaU, g2d);
    unsigned N = 1;
    double eps = 1e10, eps_old=2e10;
    std::vector<thrust::host_vector<double> > begin(2); 
    begin[0] = zeta_init, begin[1] = eta_init;
    //now we have the starting values 
    std::vector<thrust::host_vector<double> > end(begin), temp(begin);
    unsigned sizeU = u_vec.size(), sizeV = zeta_init.size();
    unsigned size2d = sizeU*sizeV;
    zeta.resize(size2d), eta.resize(size2d);
    double u0=0, u1 = u_vec[0];
    thrust::host_vector<double> zeta_old(zeta), zeta_diff( zeta), eta_old(eta), eta_diff(eta);
    while( (eps < eps_old || eps > 1e-6) && eps > 1e-7)
    {
        zeta_old = zeta, eta_old = eta; eps_old = eps; 
        temp = begin;
        //////////////////////////////////////////////////
        for( unsigned i=0; i<sizeU; i++)
        {
            u0 = i==0?0:u_vec[i-1], u1 = u_vec[i];
            dg::stepperRK17( inter, temp, end, u0, u1, N);
            for( unsigned j=0; j<sizeV; j++)
            {
                 unsigned idx = j*sizeU+i;
                 zeta[idx] = end[0][j], eta[idx] = end[1][j];
            }
            temp = end;
        }
        dg::blas1::axpby( 1., zeta, -1., zeta_old, zeta_diff);
        dg::blas1::axpby( 1., eta, -1., eta_old, eta_diff);
        dg::blas1::pointwiseDot( zeta_diff, zeta_diff, zeta_diff);
        dg::blas1::pointwiseDot( 1., eta_diff, eta_diff, 1., zeta_diff);
        eps = sqrt( dg::blas1::dot( zeta_diff, zeta_diff)/sizeU/sizeV); 
        std::cout << "Effective Absolute diff error is "<<eps<<" with "<<N<<" steps\n"; 
        N*=2;
    }

}

template< class Geometry, class container>
void transform( 
        const container& u_zeta, 
        const container& u_eta,
        thrust::host_vector<double>& u_x, 
        thrust::host_vector<double>& u_y, 
        const Geometry& g2d)
{
    u_x.resize( u_zeta.size()), u_y.resize( u_zeta.size());
    thrust::host_vector<double> uh_zeta, uh_eta;
    dg::blas1::transfer( u_zeta, uh_zeta);
    dg::blas1::transfer( u_eta, uh_eta);
    dg::blas1::pointwiseDot( uh_zeta, g2d.xr(), u_x);
    dg::blas1::pointwiseDot( 1., uh_eta, g2d.yr(), 1., u_x);
    dg::blas1::pointwiseDot( uh_zeta, g2d.xz(), u_y);
    dg::blas1::pointwiseDot( 1., uh_eta, g2d.yz(), 1., u_y);
}

}//namespace detail

//container must be compliant in blas1::transfer function
template <class IMatrix, class Matrix, class container>
struct Hector
{
    template< class Psi, class PsiX, class PsiY, class LaplacePsi>
    Hector( Psi psi, PsiX psiX, PsiY psiY, LaplacePsi laplacePsi, double psi0, double psi1, double X0, double Y0, unsigned n = 13, unsigned Nx = 2, unsigned Ny = 10, double eps_u = 1e-10) : 
        g2d_(psi, psiX, psiY, laplacePsi, psi0, psi1, X0, Y0, n, Nx, Ny, dg::DIR)
    {
        //first construct u_
        container u = construct_grid_and_u( psi, psiX, psiY, laplacePsi, psi0, psi1, X0, Y0, n, Nx, Ny, eps_u );
        //now compute u_zeta and u_eta 
        Matrix dzeta = dg::create::dx( g2d_, dg::DIR);
        Matrix deta = dg::create::dy( g2d_, dg::PER);
        container u_zeta(u), u_eta(u);
        dg::blas2::symv( dzeta, u, u_zeta);
        dg::blas1::plus( u_zeta, 1.);
        dg::blas2::symv( deta, u, u_eta);

        //we actually don't need u but it makes a good testcase 
        container zeta;
        dg::blas1::transfer(dg::evaluate( dg::coo1, g2d_), zeta);
        dg::blas1::axpby( +1., zeta, 1.,  u); //u = \tilde u + \zeta

        //now compute ZetaU and EtaU
        container a2;
        dg::blas1::transfer( g2d_.g_yy(), a2);
        dg::blas1::pointwiseDivide( a2, g2d_.g_xx(), a2); // a^2=g^ee/g^zz
        container zetaU=u_zeta, etaU=u_eta;
        dg::blas1::pointwiseDot( zetaU, zetaU, zetaU); //u_z*u_z
        dg::blas1::pointwiseDot(  etaU,  etaU,  etaU); //u_e*u_e
        dg::blas1::pointwiseDot( 1., etaU, a2, 1., zetaU); //u_z*u_z+a^2u_e*u_e
        container den( zetaU); //denominator
        dg::blas1::pointwiseDivide( u_zeta, den, zetaU); //u_z / denominator
        dg::blas1::pointwiseDivide(  u_eta, den,  etaU); 
        dg::blas1::pointwiseDot( a2, etaU, etaU); //a^2*u_e / denominator
        //now compute etaV and its inverse
        container etaV(etaU), ones(etaU.size(), 1.);
        dg::blas1::pointwiseDivide( ones, u_zeta, etaV);
        thrust::host_vector<double> etaVinv_h;
        dg::blas1::transfer( u_zeta, etaVinv_h);

        //construct c0 and scale all vector components with it
        c0_ = detail::construct_c0( etaVinv_h, g2d_);
        dg::blas1::scal(  etaV, 1./c0_);
        dg::blas1::scal( zetaU, 1./c0_);
        dg::blas1::scal(  etaU, 1./c0_);
        dg::blas1::scal( u_zeta, c0_);
        dg::blas1::scal(  u_eta, c0_);
        dg::blas1::scal(  u, c0_);
        //transfer to host
        dg::blas1::transfer( u, u_);
        detail::transform( u_zeta, u_eta, ux_, uy_, g2d_);
        dg::blas1::transfer( etaV, etaV_);
        dg::blas1::transfer( etaU, etaU_);
        dg::blas1::transfer( zetaU, zetaU_);
        //std::cout << "c0 is "<<c0_<<"\n";
    }
    double lu() const {return c0_*g2d_.lx();}
    double lv() const {return 2.*M_PI;}
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

    const orthogonal::RingGrid2d<container>& orthogonal_grid() const {return g2d_;}
    private:
    template< class Psi, class PsiX, class PsiY, class LaplacePsi>
    container construct_grid_and_u( Psi psi, PsiX psiX, PsiY psiY, LaplacePsi laplacePsi, double psi0, double psi1, double X0, double Y0, unsigned n, unsigned Nx, unsigned Ny, double eps_u ) 
    {
        //first find u( \zeta, \eta)
        double eps = 1e10, eps_old = 2e10;
        orthogonal::RingGrid2d<container> g2d_old(psi, psiX, psiY, laplacePsi, psi0, psi1, X0, Y0, n, Nx, Ny, dg::DIR);
        dg::Elliptic<orthogonal::RingGrid2d<container>, Matrix, container> ellipticD_old( g2d_old, dg::DIR, dg::PER, dg::not_normed, dg::centered);

        container u_old = dg::evaluate( dg::zero, g2d_old), u(u_old);
        container lapu = g2d_old.lapx();
        dg::Invert<container > invert_old( u_old, n*n*Nx*Ny, eps_u);
        unsigned number = invert_old( ellipticD_old, u_old, lapu);
        while( (eps < eps_old||eps > 1e-7) && eps > eps_u)
        {
            eps = eps_old;
            Nx*=2, Ny*=2;
            orthogonal::RingGrid2d<container> g2d(psi, psiX, psiY, laplacePsi, psi0, psi1, X0, Y0, n, Nx, Ny, dg::DIR);
            dg::Elliptic<orthogonal::RingGrid2d<container>, Matrix, container> ellipticD( g2d, dg::DIR, dg::PER, dg::not_normed, dg::centered);
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
    orthogonal::RingGrid2d<container> g2d_;

};

}//namespace hector
