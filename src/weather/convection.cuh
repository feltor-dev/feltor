#pragma once

#include <exception>

#include "dg/xspacelib.cuh"
#include "dg/cg.cuh"

#include "dg/timer.cuh"


struct Source
{
    __host__ __device__
    double operator()( double x, double y)
    {
        if( x<0) return x*y;
        return x*(1.-y);
    }

};
struct Params
{
    double eps, R_l, L, P, R, zeta;
};
struct Fail : public std::exception
{

    Fail( double eps): eps( eps) {}
    double epsilon() const { return eps;}
    char const* what() const throw(){ return "Failed to converge";}
  private:
    double eps;
};

template< class container>
struct Convection
{
    typedef typename container::value_type value_type;
    typedef dg::DMatrix Matrix; 
    Convection( const dg::Grid<value_type>&, Params, double eps_lap);

    void operator()( const std::vector<container>& y, std::vector<container>& yp);
    /**
     * @brief Return the normalized negative laplacian used by this object
     *
     * @return cusp matrix
     */
    const Matrix& laplacianM( ) const { return laplaceM_dir;}
    /**
     * @brief Returns phi that belong to the last y in operator()
     *
     * In a multistep scheme this belongs to the point HEAD-1
     * @return phi[0] is the electron and phi[1] the generalized ion potential
     */
    const container& potential( ) const { return phi;}
    const container& background() const { return background_;}
    const container& source() const { return source_;}
    
  private:
    const container& compute_phi( const container& omega);

    Matrix dx_per, dy_dir, dy_neu;
    Matrix laplaceM;
    Matrix laplaceM_dir;
    Matrix laplaceM_neu;


    container phi, phi_old, dxphi, dyphi;
    container dxT, source_;
    container temp;
    const container background_;
    const container w2d, v2d;
    dg::CG<container > pcg;
    //dg::ArakawaX<container> arakawaX; 

    value_type eps_, R_, R_l_, L_, P_;
    value_type eps_lap;

};

template <class container>
Convection<container>::Convection( const dg::Grid<value_type>& g, Params p, double eps_lap ): 
    phi( g.size()), phi_old(phi), dxphi( phi), dyphi( phi),
    dxT( phi), source_(phi), temp( phi),
    background_( dg::evaluate( dg::LinearY( -p.R, p.R*p.zeta), g)), 
    w2d( dg::create::w2d(g)), v2d( dg::create::v2d(g)),
    pcg( temp, g.size()), 
    //arakawaX( g), 
    eps_( p.eps), R_(p.R), R_l_( p.R_l), L_(p.L), P_(p.P), 
    eps_lap( eps_lap)
{
    dx_per = dg::create::dx( g, dg::PER, dg::XSPACE);
    dy_dir = dg::create::dy( g, dg::DIR, dg::XSPACE);
    dy_neu = dg::create::dy( g, dg::NEU, dg::XSPACE);
    laplaceM = dg::create::laplacianM( g, dg::PER, dg::DIR, dg::not_normed, dg::XSPACE, dg::symmetric);
    laplaceM_dir = dg::create::laplacianM( g, dg::PER, dg::DIR, dg::normed, dg::XSPACE, dg::symmetric);
    laplaceM_neu = dg::create::laplacianM( g, dg::PER, dg::NEU, dg::normed, dg::XSPACE, dg::symmetric);
}

template<class container>
const container& Convection<container>::compute_phi( const container& omega)
{
    dg::blas1::axpby( 2., phi, -1., phi_old);
    phi.swap( phi_old);
    dg::blas2::symv( w2d, omega, temp);
#ifdef DG_BENCHMARK
    dg::Timer t;
    t.tic();
#endif //DG_BENCHMARK
    unsigned number = pcg( laplaceM, phi, temp, v2d, eps_lap);
    if( number == pcg.get_max())
        throw Fail( eps_lap);
#ifdef DG_BENCHMARK
    std::cout << "# of pcg iterations for phi \t"<< number << "\t";
    t.toc();
    std::cout<< "took \t"<<t.diff()<<"s\n";
#endif //DG_BENCHMARK
    return phi;
}

template<class container>
void Convection<container>::operator()( const std::vector<container>& y, std::vector<container>& yp)
{
    phi = compute_phi(y[2]);
    dg::blas2::gemv( dx_per, phi, dxphi);
    dg::blas2::gemv( dy_dir, phi, dyphi);

    //arakawaX( y[0], phi, yp[0]);
    //arakawaX( y[2], phi, yp[2]);
    //first terms in Arakawa bracket
    dg::blas2::gemv( dx_per, y[0], dxT);
    dg::blas1::pointwiseDot( dxT, dyphi, yp[0]);
    dg::blas2::gemv( dx_per, y[1], temp);
    dg::blas1::pointwiseDot( temp, dyphi, yp[1]);
    dg::blas2::gemv( dx_per, y[2], temp);
    dg::blas1::pointwiseDot( temp, dyphi, yp[2]);
    //second terms in Arakawa bracket
    dg::blas2::gemv( dy_dir, y[0], temp);
    dg::blas1::pointwiseDot( temp, dxphi, temp);
    dg::blas1::axpby( -1, temp, 1, yp[0]);
    dg::blas2::gemv( dy_neu, y[1], temp);
    dg::blas1::pointwiseDot( temp, dxphi, temp);
    dg::blas1::axpby( -1, temp, 1, yp[1]);
    dg::blas2::gemv( dy_dir, y[2], temp);
    dg::blas1::pointwiseDot( temp, dxphi, temp);
    dg::blas1::axpby( -1, temp, 1, yp[2]);

    //linear terms
    dg::blas1::axpby( R_, dxphi, 1, yp[0]);
    dg::blas1::axpby( -P_,  dxT, 1, yp[2]);
    //diffusive terms
    dg::blas2::gemv( laplaceM_dir, y[0], temp);
    dg::blas1::axpby( -1., temp, 1.,yp[0]);
    dg::blas2::gemv( laplaceM_neu, y[1], temp);
    dg::blas1::axpby( -L_, temp, 1.,yp[1]);
    dg::blas2::gemv( laplaceM_dir, y[2], temp);
    dg::blas1::axpby( -P_, temp, 1.,yp[2]);
    //source term
    dg::blas1::axpby( 1., background_, 1., y[0], temp);
    thrust::transform( temp.begin(), temp.end(), y[1].begin(), source_.begin(), Source());
    dg::blas1::axpby( -R_l_*eps_*L_, source_, 1., yp[0]);
    dg::blas1::axpby( eps_*L_, source_, 1., yp[1]);
    
}

