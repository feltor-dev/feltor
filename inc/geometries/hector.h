#pragma once

#include <vector>
#include "dg/backend/grid.h"
#include "dg/backend/interpolation.cuh"
#include "dg/geometry/geometry.h"
#include "dg/elliptic.h"
#include "fluxfunctions.h"
#include "dg/cg.h"
#include "flux.h"
#include "adaption.h"



namespace dg
{
namespace geo
{

///@cond
namespace detail
{

//interpolate the two components of a vector field
struct Interpolate
{
    Interpolate( const thrust::host_vector<double>& fZeta, 
                 const thrust::host_vector<double>& fEta, 
                 const dg::aTopology2d& g2d ): 
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
    dg::Grid2d g_;
    double zeta1_, eta1_;
};

//compute c_0 
double construct_c0( const thrust::host_vector<double>& etaVinv, const dg::aTopology2d& g2d) 
{
    //this is a normal integration:
    thrust::host_vector<double> etaVinvL( dg::create::forward_transform(  etaVinv, g2d) );
    dg::Grid1d g1d( 0., 2.*M_PI, g2d.n(), g2d.Ny());
    dg::HVec eta = dg::evaluate(dg::cooX1d, g1d);
    dg::HVec w1d = dg::create::weights( g1d);
    dg::HVec int_etaVinv(eta);
    for( unsigned i=0; i<eta.size(); i++) 
        int_etaVinv[i] = interpolate( 0., eta[i], etaVinvL, g2d);
    double c0 = 2.*M_PI/dg::blas1::dot( w1d, int_etaVinv );
    return c0;

    //the following is too naiv (gives a slightly wrong result):
    //the right way to do it is to integrate de/de=1, dv/de = f(e) because in only dv/de = f(e) our integrator assumes dv/de=f(v)
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
        const dg::aTopology2d& g2d
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
        const dg::aGeometry2d& g2d
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
        //std::cout << "Effective Absolute diff error is "<<eps<<" with "<<N<<" steps\n"; 
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
    dg::SparseTensor<thrust::host_vector<double> > jac = g2d.jacobian();
    dg::blas1::pointwiseDot( uh_zeta, jac.value(0,0), u_x);
    dg::blas1::pointwiseDot( 1., uh_eta, jac.value(1,0) , 1., u_x);
    dg::blas1::pointwiseDot( uh_zeta, jac.value(0,1), u_y);
    dg::blas1::pointwiseDot( 1., uh_eta, jac.value(1,1), 1., u_y);
}

}//namespace detail
///@endcond

/**
 * @brief The High PrEcision Conformal grid generaTOR 
 *
 * @ingroup generators_geo
 * @tparam IMatrix The interpolation matrix type
 * @copydoc hide_matrix
 * @copydoc hide_container
 */
template <class IMatrix = dg::IHMatrix, class Matrix = dg::HMatrix, class container = dg::HVec>
struct Hector : public aGenerator2d
{
    /**
     * @brief Construct a conformal grid 
     *
     * @param psi \f$ \psi(x,y)\f$ the flux function and its derivatives in Cartesian coordinates (x,y)
     * @param psi0 first boundary 
     * @param psi1 second boundary
     * @param X0 a point in the inside of the ring bounded by psi0 (shouldn't be the O-point)
     * @param Y0 a point in the inside of the ring bounded by psi0 (shouldn't be the O-point)
     * @param n number of polynomials used for the internal grid
     * @param Nx initial number of points in zeta for the internal grid
     * @param Ny initial number of points in eta for the internal grid
     * @param eps_u the accuracy of u
     * @param verbose If true convergence details are printed to std::cout
     */
    Hector( const BinaryFunctorsLvl2& psi, double psi0, double psi1, double X0, double Y0, unsigned n = 13, unsigned Nx = 2, unsigned Ny = 10, double eps_u = 1e-10, bool verbose=false) : 
        g2d_(dg::geo::RibeiroFluxGenerator(psi, psi0, psi1, X0, Y0,1), n, Nx, Ny, dg::DIR)
    {
        //first construct u_
        container u = construct_grid_and_u( dg::geo::Constant(1), dg::geo::detail::LaplacePsi(psi), psi0, psi1, X0, Y0, n, Nx, Ny, eps_u , verbose);
        construct( u, psi0, psi1, dg::geo::Constant(1.), dg::geo::Constant(0.), dg::geo::Constant(1.) );
        conformal_=orthogonal_=true;
        ////we actually don't need u_ but it makes a good testcase 
        //container psi__;
        //dg::blas1::transfer(dg::pullback( psi, g2d_), psi__);
        //dg::blas1::axpby( +1., psi__, 1.,  u); //u = c0(\tilde u + \psi-\psi_0)
        //dg::blas1::plus( u,-psi0);
        //dg::blas1::scal( u, c0_);
        //dg::blas1::transfer( u, u_);
    }

    /**
     * @brief Construct an orthogonal grid with adaption
     *
     * @param psi \f$ \psi(x,y)\f$ the flux function and its derivatives in Cartesian coordinates (x,y)
     * @param chi \f$ \chi(x,y)\f$  the adaption function and its derivatives in Cartesian coordinates (x,y)
     * @param psi0 first boundary 
     * @param psi1 second boundary
     * @param X0 a point in the inside of the ring bounded by psi0 (shouldn't be the O-point)
     * @param Y0 a point in the inside of the ring bounded by psi0 (shouldn't be the O-point)
     * @param n number of polynomials used for the internal grid
     * @param Nx initial number of points in zeta for the internal grid
     * @param Ny initial number of points in eta for the internal grid
     * @param eps_u the accuracy of u
     * @param verbose If true convergence details are printed to std::cout
     */
    Hector( const BinaryFunctorsLvl2& psi, const BinaryFunctorsLvl1& chi, double psi0, double psi1, double X0, double Y0, unsigned n = 13, unsigned Nx = 2, unsigned Ny = 10, double eps_u = 1e-10, bool verbose=false) : 
        g2d_(dg::geo::RibeiroFluxGenerator(psi, psi0, psi1, X0, Y0,1), n, Nx, Ny, dg::DIR)
    {
        dg::geo::detail::LaplaceAdaptPsi lapAdaPsi( psi, chi);
        //first construct u_
        container u = construct_grid_and_u( chi.f(), lapAdaPsi, psi0, psi1, X0, Y0, n, Nx, Ny, eps_u , verbose);
        construct( u, psi0, psi1, chi.f(),dg::geo::Constant(0), chi.f() );
        orthogonal_=true;
        conformal_=false;
        ////we actually don't need u_ but it makes a good testcase 
        //container psi__;
        //dg::blas1::transfer(dg::pullback( psi, g2d_), psi__);
        //dg::blas1::axpby( +1., psi__, 1.,  u); //u = c0(\tilde u + \psi-\psi_0)
        //dg::blas1::plus( u,-psi0);
        //dg::blas1::scal( u, c0_);
        //dg::blas1::transfer( u, u_);
    }

    /**
     * @brief Construct a curvilinear grid with monitor metric
     *
     * @param psi the flux function \f$ \psi(x,y)\f$ and its derivatives in Cartesian coordinates (x,y)
      @param chi the symmetric adaption tensor \f$\chi(x,y)\f$ and its divergence in Cartesian coordinates (x,y)
     * @param psi0 first boundary 
     * @param psi1 second boundary
     * @param X0 a point in the inside of the ring bounded by psi0 (shouldn't be the O-point)
     * @param Y0 a point in the inside of the ring bounded by psi0 (shouldn't be the O-point)
     * @param n number of polynomials used for the internal grid
     * @param Nx initial number of points in zeta for the internal grid
     * @param Ny initial number of points in eta for the internal grid
     * @param eps_u the accuracy of u
     * @param verbose If true convergence details are printed to std::cout
     */
    Hector( const BinaryFunctorsLvl2& psi,const BinarySymmTensorLvl1& chi,
            double psi0, double psi1, double X0, double Y0, unsigned n = 13, unsigned Nx = 2, unsigned Ny = 10, double eps_u = 1e-10, bool verbose=false) : 
        g2d_(dg::geo::RibeiroFluxGenerator(psi, psi0, psi1, X0, Y0,1), n, Nx, Ny, dg::DIR)
    {
        //first construct u_
        container u = construct_grid_and_u( psi, chi, 
                psi0, psi1, X0, Y0, n, Nx, Ny, eps_u , verbose);
        construct( u, psi0, psi1, chi.xx(), chi.xy(), chi.yy());
        orthogonal_=conformal_=false;
        ////we actually don't need u_ but it makes a good testcase 
        //container psi__;
        //dg::blas1::transfer(dg::pullback( psi, g2d_), psi__);
        //dg::blas1::axpby( +1., psi__, 1.,  u); //u = c0(\tilde u + \psi-\psi_0)
        //dg::blas1::plus( u,-psi0);
        //dg::blas1::scal( u, c0_);
        //dg::blas1::transfer( u, u_);
    }


    /**
     * @brief Return the internally used orthogonal grid
     *
     * @return  orthogonal zeta, eta grid
     */
    const dg::geo::CurvilinearGrid2d& internal_grid() const {return g2d_;}
    virtual Hector* clone() const{return new Hector(*this);}
    bool isConformal() const {return conformal_;}
    private:
    virtual double do_width() const {return lu_;}
    virtual double do_height() const {return 2.*M_PI;}
    /**
     * @brief True if orthogonal constructor was used
     *
     * @return true if orthogonal constructor was used
     */
    virtual bool do_isOrthogonal() const {return orthogonal_;}
    virtual void do_generate( const thrust::host_vector<double>& u1d, 
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

        dg::blas2::symv( Q, g2d_.map()[0], x);
        dg::blas2::symv( Q, g2d_.map()[1], y);
        dg::blas2::symv( Q, ux_, ux);
        dg::blas2::symv( Q, uy_, uy);
        dg::blas2::symv( Q, vx_, vx);
        dg::blas2::symv( Q, vy_, vy);
        ////Test if u1d is u
        //thrust::host_vector<double> u(u1d.size()*v1d.size());
        //dg::blas2::symv( Q, u_, u);
        //dg::HVec u2d(u1d.size()*v1d.size());
        //for( unsigned i=0; i<v1d.size(); i++)
        //    for( unsigned j=0; j<u1d.size(); j++)
        //        u2d[i*u1d.size()+j] = u1d[j];
        //dg::blas1::axpby( 1., u2d, -1., u);
        //double eps = dg::blas1::dot( u,u);
        //std::cout << "Error in u is "<<eps<<std::endl;
    }

    container construct_grid_and_u( const aBinaryFunctor& chi, const aBinaryFunctor& lapChiPsi, double psi0, double psi1, double X0, double Y0, unsigned n, unsigned Nx, unsigned Ny, double eps_u , bool verbose) 
    {
        //first find u( \zeta, \eta)
        double eps = 1e10, eps_old = 2e10;
        dg::geo::CurvilinearGrid2d g2d_old = g2d_;
        container adapt = dg::pullback(chi, g2d_old);
        dg::Elliptic<dg::geo::CurvilinearGrid2d, Matrix, container> ellipticD_old( g2d_old, dg::DIR, dg::PER, dg::not_normed, dg::centered);
        ellipticD_old.set_chi( adapt);

        container u_old = dg::evaluate( dg::zero, g2d_old), u(u_old);
        container lapu = dg::pullback( lapChiPsi, g2d_old);
        dg::Invert<container > invert_old( u_old, n*n*Nx*Ny, eps_u);
        unsigned number = invert_old( ellipticD_old, u_old, lapu);
        while( (eps < eps_old||eps > 1e-7) && eps > eps_u)
        {
            eps = eps_old;
            g2d_.multiplyCellNumbers(2,2);
            if(verbose) std::cout << "Nx "<<Nx<<" Ny "<<Ny<<std::flush;
            dg::Elliptic<dg::geo::CurvilinearGrid2d, Matrix, container> ellipticD( g2d_, dg::DIR, dg::PER, dg::not_normed, dg::centered);
            adapt = dg::pullback(chi, g2d_);
            ellipticD.set_chi( adapt);
            lapu = dg::pullback( lapChiPsi, g2d_);
            const container vol2d = dg::create::weights( g2d_);
            const IMatrix Q = dg::create::interpolation( g2d_, g2d_old);
            u = dg::evaluate( dg::zero, g2d_);
            container u_diff( u);
            dg::blas2::gemv( Q, u_old, u_diff);

            dg::Invert<container > invert( u_diff, n*n*Nx*Ny, 0.1*eps_u);
            number = invert( ellipticD, u, lapu);
            dg::blas1::axpby( 1. ,u, -1., u_diff);
            eps = sqrt( dg::blas2::dot( u_diff, vol2d, u_diff) / dg::blas2::dot( u, vol2d, u) );
            if(verbose) std::cout <<" iter "<<number<<" error "<<eps<<"\n";
            g2d_old = g2d_;
            u_old = u;
            number++;//get rid of warning
        }
        return u;
    }

    container construct_grid_and_u( const BinaryFunctorsLvl2& psi, 
            const BinarySymmTensorLvl1& chi, double psi0, double psi1, double X0, double Y0, unsigned n, unsigned Nx, unsigned Ny, double eps_u, bool verbose ) 
    {
        dg::geo::detail::LaplaceChiPsi lapChiPsi( psi, chi);
        //first find u( \zeta, \eta)
        double eps = 1e10, eps_old = 2e10;
        dg::geo::CurvilinearGrid2d g2d_old = g2d_;
        dg::TensorElliptic<dg::geo::CurvilinearGrid2d, Matrix, container> ellipticD_old( g2d_old, dg::DIR, dg::PER, dg::not_normed, dg::centered);
        ellipticD_old.transform_and_set( chi.xx(), chi.xy(), chi.yy());

        container u_old = dg::evaluate( dg::zero, g2d_old), u(u_old);
        container lapu = dg::pullback( lapChiPsi, g2d_old);
        dg::Invert<container > invert_old( u_old, n*n*Nx*Ny, eps_u);
        unsigned number = invert_old( ellipticD_old, u_old, lapu);
        while( (eps < eps_old||eps > 1e-7) && eps > eps_u)
        {
            eps = eps_old;
            g2d_.multiplyCellNumbers(2,2);
            if(verbose)std::cout << "Nx "<<Nx<<" Ny "<<Ny<<std::flush;
            dg::TensorElliptic<dg::geo::CurvilinearGrid2d, Matrix, container> ellipticD( g2d_, dg::DIR, dg::PER, dg::not_normed, dg::centered);
            ellipticD.transform_and_set( chi.xx(), chi.xy(), chi.yy() );
            lapu = dg::pullback( lapChiPsi, g2d_);
            const container vol2d = dg::create::weights( g2d_);
            const IMatrix Q = dg::create::interpolation( g2d_, g2d_old);
            u = dg::evaluate( dg::zero, g2d_);
            container u_diff( u);
            dg::blas2::gemv( Q, u_old, u_diff);

            dg::Invert<container > invert( u_diff, n*n*Nx*Ny, 0.1*eps_u);
            number = invert( ellipticD, u, lapu);
            dg::blas1::axpby( 1. ,u, -1., u_diff);
            eps = sqrt( dg::blas2::dot( u_diff, vol2d, u_diff) / dg::blas2::dot( u, vol2d, u) );
            if(verbose) std::cout <<" iter "<<number<<" error "<<eps<<"\n";
            g2d_old = g2d_;
            u_old = u;
            number++;//get rid of warning
        }
        return u;
    }

    void construct(const container& u, double psi0, double psi1, const aBinaryFunctor& chi_XX, const aBinaryFunctor& chi_XY, const aBinaryFunctor& chi_YY)
    {
        //now compute u_zeta and u_eta 
        Matrix dzeta = dg::create::dx( g2d_, dg::DIR);
        Matrix deta = dg::create::dy( g2d_, dg::PER);
        container u_zeta(u), u_eta(u);
        dg::blas2::symv( dzeta, u, u_zeta);
        dg::blas1::plus( u_zeta, (psi1-psi0)/g2d_.lx());
        dg::blas2::symv( deta, u, u_eta);


        thrust::host_vector<double> chi_ZZ, chi_ZE, chi_EE;
        dg::pushForwardPerp( chi_XX, chi_XY, chi_YY, chi_ZZ, chi_ZE, chi_EE, g2d_);
        container chiZZ, chiZE, chiEE;
        dg::blas1::transfer( chi_ZZ, chiZZ);
        dg::blas1::transfer( chi_ZE, chiZE);
        dg::blas1::transfer( chi_EE, chiEE);

        //now compute ZetaU and EtaU
        container temp_zeta(u), temp_eta(u);
        dg::blas1::pointwiseDot( chiZZ, u_zeta, temp_zeta);
        dg::blas1::pointwiseDot( 1. ,chiZE, u_eta, 1., temp_zeta);
        dg::blas1::pointwiseDot( chiZE, u_zeta, temp_eta);
        dg::blas1::pointwiseDot( 1. ,chiEE, u_eta, 1., temp_eta);
        container temp_scalar(u);
        dg::blas1::pointwiseDot( u_zeta, temp_zeta, temp_scalar);
        dg::blas1::pointwiseDot( 1., u_eta, temp_eta, 1., temp_scalar);
        container zetaU=temp_zeta, etaU=temp_eta;
        dg::blas1::pointwiseDivide( zetaU, temp_scalar, zetaU); 
        dg::blas1::pointwiseDivide(  etaU, temp_scalar,  etaU); 
        //now compute etaV and its inverse
        container etaVinv(u_zeta), etaV(etaVinv);
        dg::blas1::pointwiseDot( etaVinv, chiZZ, etaVinv);
        dg::SparseElement<container> perp_vol = dg::tensor::determinant(g2d_.metric());
        dg::tensor::sqrt(perp_vol);
        dg::tensor::invert(perp_vol);
        dg::tensor::pointwiseDot( etaVinv, perp_vol, etaVinv);
        dg::blas1::transform( etaVinv, etaV, dg::INVERT<double>());
        thrust::host_vector<double> etaVinv_h;
        dg::blas1::transfer( etaVinv, etaVinv_h);
        //now compute v_zeta and v_eta
        container v_zeta(u), v_eta(u);
        dg::blas1::axpby( -1., temp_eta, 0.,v_zeta);
        dg::blas1::axpby( +1., temp_zeta, 0.,v_eta);
        dg::tensor::pointwiseDot( v_zeta, perp_vol, v_zeta);
        dg::tensor::pointwiseDot( v_eta, perp_vol, v_eta);

        //construct c0 and scale all vector components with it
        c0_ = fabs( detail::construct_c0( etaVinv_h, g2d_));
        if( psi1 < psi0) c0_*=-1;
        lu_ = c0_*(psi1-psi0);
        //std::cout << "c0 is "<<c0_<<"\n";
        dg::blas1::scal(  etaV, 1./c0_);
        dg::blas1::scal( zetaU, 1./c0_);
        dg::blas1::scal(  etaU, 1./c0_);
        dg::blas1::scal( u_zeta, c0_);
        dg::blas1::scal( v_zeta, c0_);
        dg::blas1::scal(  u_eta, c0_);
        dg::blas1::scal(  v_eta, c0_);
        //transfer to host
        detail::transform( u_zeta, u_eta, ux_, uy_, g2d_);
        detail::transform( v_zeta, v_eta, vx_, vy_, g2d_);
        dg::blas1::transfer( etaV, etaV_);
        dg::blas1::transfer( etaU, etaU_);
        dg::blas1::transfer( zetaU, zetaU_);
    }
    private:
    bool conformal_, orthogonal_;
    double c0_, lu_;
    thrust::host_vector<double> u_, ux_, uy_, vx_, vy_;
    thrust::host_vector<double> etaV_, zetaU_, etaU_;
    dg::geo::CurvilinearGrid2d g2d_;

};

}//namespace geo
}//namespace dg
