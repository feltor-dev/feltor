#pragma once

#include <vector>
#include "dg/algorithm.h"
#include "fluxfunctions.h"
#include "curvilinear.h"
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
template<class real_type>
struct Interpolate
{
    Interpolate( const thrust::host_vector<real_type>& fZeta,
                 const thrust::host_vector<real_type>& fEta,
                 const dg::aTopology2d& g2d ):
        iter0_( dg::forward_transform( fZeta, g2d) ),
        iter1_( dg::forward_transform(  fEta, g2d) ),
        g_(g2d), zeta1_(g2d.x1()), eta1_(g2d.y1()){}
    void operator()(real_type, const std::array<real_type,2>& zeta, std::array<real_type,2>& fZeta)
    {
        fZeta[0] = interpolate( dg::lspace, iter0_, fmod( zeta[0]+zeta1_, zeta1_), fmod( zeta[1]+eta1_, eta1_), g_);
        fZeta[1] = interpolate( dg::lspace, iter1_, fmod( zeta[0]+zeta1_, zeta1_), fmod( zeta[1]+eta1_, eta1_), g_);
        //fZeta[0] = interpolate(  zeta[0], zeta[1], iter0_, g_);
        //fZeta[1] = interpolate(  zeta[0], zeta[1], iter1_, g_);
    }
    void operator()(real_type, const std::array<thrust::host_vector<real_type>,2 >& zeta, std::array< thrust::host_vector<real_type>,2 >& fZeta)
    {
        for( unsigned i=0; i<zeta[0].size(); i++)
        {
            fZeta[0][i] = interpolate( dg::lspace, iter0_, fmod( zeta[0][i]+zeta1_, zeta1_), fmod( zeta[1][i]+eta1_, eta1_), g_);
            fZeta[1][i] = interpolate( dg::lspace, iter1_, fmod( zeta[0][i]+zeta1_, zeta1_), fmod( zeta[1][i]+eta1_, eta1_), g_);
        }
    }
    private:
    thrust::host_vector<real_type> iter0_;
    thrust::host_vector<real_type> iter1_;
    dg::RealGrid2d<real_type> g_;
    real_type zeta1_, eta1_;
};

//compute c_0
template<class real_type>
real_type construct_c0( const thrust::host_vector<real_type>& etaVinv, const dg::aRealTopology2d<real_type>& g2d)
{
    //this is a normal integration:
    thrust::host_vector<real_type> etaVinvL( dg::forward_transform(  etaVinv, g2d) );
    dg::Grid1d g1d( 0., 2.*M_PI, g2d.ny(), g2d.Ny());
    dg::HVec eta = dg::evaluate(dg::cooX1d, g1d);
    dg::HVec w1d = dg::create::weights( g1d);
    dg::HVec int_etaVinv(eta);
    for( unsigned i=0; i<eta.size(); i++)
        int_etaVinv[i] = interpolate( dg::lspace, etaVinvL, 0., eta[i], g2d);
    real_type c0 = 2.*M_PI/dg::blas1::dot( w1d, int_etaVinv );
    return c0;

    //the following is too naiv (gives a slightly wrong result):
    //the right way to do it is to integrate de/de=1, dv/de = f(e) because in only dv/de = f(e) our integrator assumes dv/de=f(v)
    //Interpolate inter( thrust::host_vector<real_type>( etaVinv.size(), 0), etaVinv, g2d);
    //thrust::host_vector<real_type> begin( 2, 0), end(begin), end_old(begin);
    //begin[0] = 0, begin[1] = 0;
    //real_type eps = 1e10, eps_old = 2e10;
    //unsigned N = 5;
    //while( (eps < eps_old || eps > 1e-7)&& eps > 1e-12)
    //{
    //    eps_old = eps, end_old = end;
    //    N*=2; dg::stepperRK( "ARK-4-2-3 (explicit)", inter, 0., begin, 2*M_PI, end, N);
    //    eps = fabs( end[1]-end_old[1]);
    //    std::cout << "\t error eps "<<eps<<" with "<<N<<" steps: " << 2*M_PI/end[1]<<"\n";
    //    std::cout << "\t error c0  "<<fabs(c0-2.*M_PI/end[1])<<" with "<<N<<" steps: " << 2*M_PI/end[1]<<"\n";
    //}
    ////std::cout <<end_old[2] << " "<<end[2] << "error in y is "<<y_eps<<"\n";
    //real_type f_psi = 2.*M_PI/end_old[1];

    //return f_psi;
}


//compute the vector of zeta and eta - values that form first v surface
template<class real_type>
void compute_zev(
        const thrust::host_vector<real_type>& etaV,
        const thrust::host_vector<real_type>& v_vec,
        thrust::host_vector<real_type>& eta,
        const dg::aRealTopology2d<real_type>& g2d
        )
{
    Interpolate<real_type> iter( thrust::host_vector<real_type>( etaV.size(),
                0), etaV, g2d);
    eta.resize( v_vec.size());
    thrust::host_vector<real_type> eta_old(v_vec.size(), 0), eta_diff( eta_old);
    std::array<real_type,2> begin{ 0, 0}, end(begin), temp(begin);
    begin[0] = 0., begin[1] = 0.;
    unsigned steps = 1;
    real_type eps = 1e10, eps_old=2e10;
    using Vec = std::array<double,2>;
    dg::SinglestepTimeloop<Vec> odeint( dg::RungeKutta<Vec>( "Feagin-17-8-10",
                begin), iter);
    while( (eps < eps_old||eps > 1e-7) && eps > 1e-12)
    {
        //begin is left const
        eps_old = eps, eta_old = eta;
        odeint.integrate_steps( 0, begin, v_vec[0], end, steps);
        eta[0] = end[1];
        for( unsigned i=1; i<v_vec.size(); i++)
        {
            temp = end;
            odeint.integrate_steps( v_vec[i-1], temp, v_vec[i], end, steps);
            eta[i] = end[1];
        }
        temp = end;
        odeint.integrate_steps( v_vec[v_vec.size()-1], begin, 2.*M_PI, end, steps);
        dg::blas1::axpby( 1., eta, -1., eta_old, eta_diff);
        eps =  dg::fast_l2norm( eta_diff)/dg::fast_l2norm( eta);
        //std::cout << "rel. error is "<<eps<<" with "<<steps<<" steps\n";
        //std::cout << "abs. error is "<<( 2.*M_PI-end[1])<<"\n";
        steps*=2;
    }
}

template<class real_type>
void construct_grid(
        const thrust::host_vector<real_type>& zetaU, //2d Zeta component
        const thrust::host_vector<real_type>& etaU,  //2d Eta component
        const thrust::host_vector<real_type>& u_vec,  //1d u values
        const thrust::host_vector<real_type>& zeta_init, //1d intial values
        const thrust::host_vector<real_type>& eta_init, //1d intial values
        thrust::host_vector<real_type>& zeta,
        thrust::host_vector<real_type>& eta,
        const dg::aRealGeometry2d<real_type>& g2d
    )
{
    Interpolate<real_type> inter( zetaU, etaU, g2d);
    unsigned N = 1;
    real_type eps = 1e10, eps_old=2e10;
    std::array<thrust::host_vector<real_type>,2 > begin;
    begin[0] = zeta_init, begin[1] = eta_init;
    //now we have the starting values
    std::array<thrust::host_vector<real_type>,2 > end(begin), temp(begin);
    unsigned sizeU = u_vec.size(), sizeV = zeta_init.size();
    unsigned size2d = sizeU*sizeV;
    zeta.resize(size2d), eta.resize(size2d);
    real_type u0=0, u1 = u_vec[0];
    thrust::host_vector<real_type> zeta_old(zeta), zeta_diff( zeta), eta_old(eta), eta_diff(eta);
    using Vec = std::array<thrust::host_vector<real_type>,2>;
    dg::SinglestepTimeloop<Vec> odeint( dg::RungeKutta<Vec>( "Feagin-17-8-10",
                temp), inter);
    while( (eps < eps_old || eps > 1e-6) && eps > 1e-7)
    {
        zeta_old = zeta, eta_old = eta; eps_old = eps;
        temp = begin;
        //////////////////////////////////////////////////
        for( unsigned i=0; i<sizeU; i++)
        {
            u0 = i==0 ? 0 : u_vec[i-1], u1 = u_vec[i];
            odeint.integrate_steps( u0, temp, u1, end, N);
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
    dg::assign( u_zeta, uh_zeta);
    dg::assign( u_eta, uh_eta);
    dg::SparseTensor<thrust::host_vector<double> > jac = g2d.jacobian();
    dg::tensor::multiply2d( jac.transpose(), uh_zeta, uh_eta, u_x, u_y);
}

}//namespace detail
///@endcond

/**
 * @brief The High PrEcision Conformal grid generaTOR
 *
 * @note implements the algorithm described in <a href =
 * "https://doi.org/10.1016/j.jcp.2017.03.056"> M. Wiesenberger, M. Held, L.
 * Einkemmer Streamline integration as a method for two-dimensional elliptic
 * grid generation Journal of Computational Physics 340, 435-450 (2017) </a>
 *
 * @snippet flux_b.cpp hector
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
    Hector( const CylindricalFunctorsLvl2& psi, double psi0, double psi1, double X0, double Y0, unsigned n = 13, unsigned Nx = 2, unsigned Ny = 10, double eps_u = 1e-10, bool verbose=false) :
        m_g2d(dg::geo::RibeiroFluxGenerator(psi, psi0, psi1, X0, Y0,1), n, Nx, Ny, dg::DIR)
    {
        //first construct m_u
        container u = construct_grid_and_u( dg::geo::Constant(1), dg::geo::detail::LaplacePsi(psi), eps_u , verbose);
        construct( u, psi0, psi1, dg::geo::Constant(1.), dg::geo::Constant(0.), dg::geo::Constant(1.), verbose);
        m_conformal=m_orthogonal=true;
        ////we actually don't need m_u but it makes a good testcase
        //container psi__;
        //dg::assign(dg::pullback( psi, m_g2d), psi__);
        //dg::blas1::axpby( +1., psi__, 1.,  u); //u = c0(\tilde u + \psi-\psi_0)
        //dg::blas1::plus( u,-psi0);
        //dg::blas1::scal( u, m_c0);
        //dg::assign( u, m_u);
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
    Hector( const CylindricalFunctorsLvl2& psi, const CylindricalFunctorsLvl1& chi, double psi0, double psi1, double X0, double Y0, unsigned n = 13, unsigned Nx = 2, unsigned Ny = 10, double eps_u = 1e-10, bool verbose=false) :
        m_g2d(dg::geo::RibeiroFluxGenerator(psi, psi0, psi1, X0, Y0,1), n, Nx, Ny, dg::DIR)
    {
        dg::geo::detail::LaplaceAdaptPsi lapAdaPsi( psi, chi);
        //first construct m_u
        container u = construct_grid_and_u( chi.f(), lapAdaPsi, eps_u , verbose);
        construct( u, psi0, psi1, chi.f(),dg::geo::Constant(0), chi.f(), verbose );
        m_orthogonal=true;
        m_conformal=false;
        ////we actually don't need m_u but it makes a good testcase
        //container psi__;
        //dg::assign(dg::pullback( psi, m_g2d), psi__);
        //dg::blas1::axpby( +1., psi__, 1.,  u); //u = c0(\tilde u + \psi-\psi_0)
        //dg::blas1::plus( u,-psi0);
        //dg::blas1::scal( u, m_c0);
        //dg::assign( u, m_u);
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
    Hector( const CylindricalFunctorsLvl2& psi,const CylindricalSymmTensorLvl1& chi,
            double psi0, double psi1, double X0, double Y0, unsigned n = 13, unsigned Nx = 2, unsigned Ny = 10, double eps_u = 1e-10, bool verbose=false) :
        m_g2d(dg::geo::RibeiroFluxGenerator(psi, psi0, psi1, X0, Y0,1), n, Nx, Ny, dg::DIR)
    {
        //first construct m_u
        container u = construct_grid_and_u( psi, chi, eps_u , verbose);
        construct( u, psi0, psi1, chi.xx(), chi.xy(), chi.yy(), verbose);
        m_orthogonal=m_conformal=false;
        ////we actually don't need m_u but it makes a good testcase
        //container psi__;
        //dg::assign(dg::pullback( psi, m_g2d), psi__);
        //dg::blas1::axpby( +1., psi__, 1.,  u); //u = c0(\tilde u + \psi-\psi_0)
        //dg::blas1::plus( u,-psi0);
        //dg::blas1::scal( u, m_c0);
        //dg::assign( u, m_u);
    }


    /**
     * @brief Return the internally used orthogonal grid
     *
     * @return  orthogonal zeta, eta grid
     */
    const dg::geo::CurvilinearGrid2d& internal_grid() const {return m_g2d;}
    virtual Hector* clone() const override final{return new Hector(*this);}
    bool isConformal() const {return m_conformal;}
    private:
    virtual double do_width() const override final {return m_lu;}
    virtual double do_height() const override final {return 2.*M_PI;}
    /**
     * @brief True if orthogonal constructor was used
     *
     * @return true if orthogonal constructor was used
     */
    virtual bool do_isOrthogonal() const override final {return m_orthogonal;}
    virtual void do_generate( const thrust::host_vector<double>& u1d,
                     const thrust::host_vector<double>& v1d,
                     thrust::host_vector<double>& x,
                     thrust::host_vector<double>& y,
                     thrust::host_vector<double>& ux,
                     thrust::host_vector<double>& uy,
                     thrust::host_vector<double>& vx,
                     thrust::host_vector<double>& vy) const override final
    {
        thrust::host_vector<double> eta_init, zeta, eta;
        detail::compute_zev( m_etaV, v1d, eta_init, m_g2d);
        thrust::host_vector<double> zeta_init( eta_init.size(), 0.);
        detail::construct_grid( m_zetaU, m_etaU, u1d, zeta_init, eta_init, zeta, eta, m_g2d);
        //the box is periodic in eta and the y=0 line needs not to coincide with the eta=0 line
        for( unsigned i=0; i<eta.size(); i++)
            eta[i] = fmod(eta[i]+2.*M_PI, 2.*M_PI);
        dg::IHMatrix Q = dg::create::interpolation( zeta, eta, m_g2d);

        dg::blas2::symv( Q, m_g2d.map()[0], x);
        dg::blas2::symv( Q, m_g2d.map()[1], y);
        dg::blas2::symv( Q, m_ux, ux);
        dg::blas2::symv( Q, m_uy, uy);
        dg::blas2::symv( Q, m_vx, vx);
        dg::blas2::symv( Q, m_vy, vy);
        ////Test if u1d is u
        //thrust::host_vector<double> u(u1d.size()*v1d.size());
        //dg::blas2::symv( Q, m_u, u);
        //dg::HVec u2d(u1d.size()*v1d.size());
        //for( unsigned i=0; i<v1d.size(); i++)
        //    for( unsigned j=0; j<u1d.size(); j++)
        //        u2d[i*u1d.size()+j] = u1d[j];
        //dg::blas1::axpby( 1., u2d, -1., u);
        //double eps = dg::blas1::dot( u,u);
        //std::cout << "Error in u is "<<eps<<std::endl;
    }

    container construct_grid_and_u( const CylindricalFunctor& chi, const CylindricalFunctor& lapChiPsi, double eps_u , bool verbose)
    {
        //first find u( \zeta, \eta)
        double eps = 1e10, eps_old = 2e10;
        dg::geo::CurvilinearGrid2d g2d_old = m_g2d;
        container adapt = dg::pullback(chi, g2d_old);
        dg::Elliptic2d<dg::geo::CurvilinearGrid2d, Matrix, container> ellipticD_old( g2d_old, dg::DIR, dg::PER, dg::centered);
        ellipticD_old.set_chi( adapt);

        container u_old = dg::evaluate( dg::zero, g2d_old), u(u_old);
        dg::PCG<container > invert_old( u_old, g2d_old.size());
        container lapu = dg::pullback( lapChiPsi, g2d_old);
        unsigned number = invert_old.solve( ellipticD_old, u_old, lapu, ellipticD_old.precond(), ellipticD_old.weights(), eps_u);
        if(verbose) std::cout << "Nx "<<m_g2d.Nx()<<" Ny "<<m_g2d.Ny()<<std::flush;
        if(verbose) std::cout <<" iter "<<number<<" error "<<eps<<"\n";
        while( (eps < eps_old||eps > 1e-7) && eps > eps_u)
        {
            eps = eps_old;
            m_g2d.multiplyCellNumbers(2,2);
            if(verbose) std::cout << "Nx "<<m_g2d.Nx()<<" Ny "<<m_g2d.Ny()<<std::flush;
            dg::Elliptic2d<dg::geo::CurvilinearGrid2d, Matrix, container> ellipticD( m_g2d, dg::DIR, dg::PER, dg::centered);
            adapt = dg::pullback(chi, m_g2d);
            ellipticD.set_chi( adapt);
            const container vol2d = dg::create::weights( m_g2d);
            const IMatrix Q = dg::create::interpolation( m_g2d, g2d_old);
            container u_diff = dg::evaluate( dg::zero, m_g2d);
            dg::blas2::gemv( Q, u_old, u_diff);
            u = u_diff;

            dg::PCG<container > invert( u_diff, m_g2d.size());
            lapu = dg::pullback( lapChiPsi, m_g2d);
            number = invert.solve( ellipticD, u, lapu, ellipticD.precond(), ellipticD.weights(), 0.1*eps_u);
            dg::blas1::axpby( 1. ,u, -1., u_diff);
            eps = sqrt( dg::blas2::dot( u_diff, vol2d, u_diff) / dg::blas2::dot( u, vol2d, u) );
            if(verbose) std::cout <<" iter "<<number<<" error "<<eps<<"\n";
            g2d_old = m_g2d;
            u_old = u;
            number++;//get rid of warning
        }
        return u;
    }

    container construct_grid_and_u( const CylindricalFunctorsLvl2& psi,
            const CylindricalSymmTensorLvl1& chi, double eps_u, bool verbose )
    {
        dg::geo::detail::LaplaceChiPsi lapChiPsi( psi, chi);
        //first find u( \zeta, \eta)
        double eps = 1e10, eps_old = 2e10;
        dg::geo::CurvilinearGrid2d g2d_old = m_g2d;
        dg::Elliptic2d<dg::geo::CurvilinearGrid2d, Matrix, container> ellipticD_old( g2d_old, dg::DIR, dg::PER, dg::centered);
        dg::SparseTensor<container> chi_t;
        dg::pushForwardPerp( chi.xx(), chi.xy(), chi.yy(), chi_t, g2d_old);

        ellipticD_old.set_chi( chi_t);

        container u_old = dg::evaluate( dg::zero, g2d_old), u(u_old);
        dg::PCG<container > invert_old( u_old, g2d_old.size());
        container lapu = dg::pullback( lapChiPsi, g2d_old);
        unsigned number = invert_old.solve( ellipticD_old, u_old, lapu, ellipticD_old.precond(), ellipticD_old.weights(), eps_u);
        while( (eps < eps_old||eps > 1e-7) && eps > eps_u)
        {
            eps = eps_old;
            m_g2d.multiplyCellNumbers(2,2);
            if(verbose) std::cout << "Nx "<<m_g2d.Nx()<<" Ny "<<m_g2d.Ny()<<std::flush;
            dg::Elliptic2d<dg::geo::CurvilinearGrid2d, Matrix, container> ellipticD( m_g2d, dg::DIR, dg::PER, dg::centered);
            dg::pushForwardPerp( chi.xx(), chi.xy(), chi.yy(), chi_t, m_g2d);

            ellipticD.set_chi( chi_t );
            const container vol2d = dg::create::weights( m_g2d);
            const IMatrix Q = dg::create::interpolation( m_g2d, g2d_old);
            container u_diff = dg::evaluate( dg::zero, m_g2d);
            dg::blas2::gemv( Q, u_old, u_diff);
            u = u_diff;

            dg::PCG<container > invert( u_diff, m_g2d.size());
            lapu = dg::pullback( lapChiPsi, m_g2d);
            number = invert.solve( ellipticD, u, lapu, ellipticD.precond(), ellipticD.weights(), 0.1*eps_u);
            dg::blas1::axpby( 1. ,u, -1., u_diff);
            eps = sqrt( dg::blas2::dot( u_diff, vol2d, u_diff) / dg::blas2::dot( u, vol2d, u) );
            if(verbose) std::cout <<" iter "<<number<<" error "<<eps<<"\n";
            g2d_old = m_g2d;
            u_old = u;
            number++;//get rid of warning
        }
        return u;
    }

    void construct(const container& u, double psi0, double psi1, const CylindricalFunctor& chi_XX, const CylindricalFunctor& chi_XY, const CylindricalFunctor& chi_YY, bool verbose)
    {
        //now compute u_zeta and u_eta
        Matrix dzeta = dg::create::dx( m_g2d, dg::DIR);
        Matrix deta = dg::create::dy( m_g2d, dg::PER);
        container u_zeta(u), u_eta(u);
        dg::blas2::symv( dzeta, u, u_zeta);
        dg::blas1::plus( u_zeta, (psi1-psi0)/m_g2d.lx());
        dg::blas2::symv( deta, u, u_eta);



        dg::SparseTensor<container> chi;
        dg::pushForwardPerp( chi_XX, chi_XY, chi_YY, chi, m_g2d);

        //now compute ZetaU and EtaU
        container temp_zeta(u), temp_eta(u);
        dg::tensor::multiply2d( chi, u_zeta, u_eta, temp_zeta, temp_eta);
        container temp_scalar(u);
        dg::blas1::pointwiseDot( 1., u_eta, temp_eta, 1., u_zeta, temp_zeta, 0., temp_scalar);
        container zetaU=temp_zeta, etaU=temp_eta;
        dg::blas1::pointwiseDivide( zetaU, temp_scalar, zetaU);
        dg::blas1::pointwiseDivide(  etaU, temp_scalar,  etaU);

        //now compute etaV and its inverse
        container etaVinv(u_zeta), etaV(etaVinv);
        container perp_vol = dg::tensor::volume(m_g2d.metric());
        dg::blas1::pointwiseDot( 1., u_zeta, perp_vol, chi.value(0,0), 0., etaVinv);
        dg::blas1::transform( etaVinv, etaV, dg::INVERT<double>());
        thrust::host_vector<double> etaVinv_h;
        dg::assign( etaVinv, etaVinv_h);
        //now compute v_zeta and v_eta
        container v_zeta(u), v_eta(u);
        dg::blas1::axpby( -1., temp_eta, 0.,v_zeta);
        dg::blas1::axpby( +1., temp_zeta, 0.,v_eta);
        dg::blas1::pointwiseDot( v_zeta, perp_vol, v_zeta);
        dg::blas1::pointwiseDot( v_eta, perp_vol, v_eta);

        //construct c0 and scale all vector components with it
        m_c0 = fabs( detail::construct_c0( etaVinv_h, m_g2d));
        if( psi1 < psi0) m_c0*=-1;
        m_lu = m_c0*(psi1-psi0);
        if(verbose) std::cout << "c0 is "<<m_c0<<"\n";
        dg::blas1::scal(  etaV, 1./m_c0);
        dg::blas1::scal( zetaU, 1./m_c0);
        dg::blas1::scal(  etaU, 1./m_c0);
        dg::blas1::scal( u_zeta, m_c0);
        dg::blas1::scal( v_zeta, m_c0);
        dg::blas1::scal(  u_eta, m_c0);
        dg::blas1::scal(  v_eta, m_c0);
        //transfer to host
        detail::transform( u_zeta, u_eta, m_ux, m_uy, m_g2d);
        detail::transform( v_zeta, v_eta, m_vx, m_vy, m_g2d);
        dg::assign( etaV, m_etaV);
        dg::assign( etaU, m_etaU);
        dg::assign( zetaU, m_zetaU);
    }
    private:
    bool m_conformal, m_orthogonal;
    double m_c0, m_lu;
    thrust::host_vector<double> m_ux, m_uy, m_vx, m_vy;
    thrust::host_vector<double> m_etaV, m_zetaU, m_etaU;
    dg::geo::CurvilinearGrid2d m_g2d;

};

}//namespace geo
}//namespace dg
