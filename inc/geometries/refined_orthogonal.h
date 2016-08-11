#pragma once

#include "dg/geometry/refined_grid.h"
#include "orthogonal.h"



namespace orthogonal
{
namespace refined
{

template< class container>
struct RingGrid2d; 

/**
 * @brief A three-dimensional grid based on "almost-conformal" coordinates by Ribeiro and Scott 2010
 */
template< class container>
struct RingGrid3d : public dg::refined::Grid3d
{
    typedef dg::OrthogonalCylindricalTag metric_category;
    typedef RingGrid2d<container> perpendicular_grid;

    /**
     * @brief Construct 
     *
     * @param gp The geometric parameters define the magnetic field
     * @param psi_0 lower boundary for psi
     * @param psi_1 upper boundary for psi
     * @param n The dG number of polynomials
     * @param Nx The number of points in x-direction
     * @param Ny The number of points in y-direction
     * @param Nz The number of points in z-direction
     * @param bcx The boundary condition in x (y,z are periodic)
     */
    RingGrid3d( unsigned multiple_x, unsigned multiple_y, solovev::GeomParameters gp, double psi_0, double psi_1, unsigned n, unsigned n_old, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx): 
        dg::refined::Grid3d( multiple_x, multiple_y, 0, 1, 0., 2.*M_PI, 0., 2.*M_PI, n, n_old, Nx, Ny, Nz, bcx, dg::PER, dg::PER),
        g_assoc_( gp, psi_0, psi_1, n_old, Nx, Ny, Nz, bcx)
    { 
        orthogonal::detail::Fpsi fpsi( gp);
        double x_1 = fpsi.find_x1( psi_0, psi_1);
        if( x_1 > 0)
            init_X_boundaries( 0., x_1);
        else
        {
            init_X_boundaries( x_1, 0.);
            std::swap( psi_0, psi_1);
        }
        //simultaneously compute one flux surface after the other
        thrust::host_vector<double> x_vec(this->n()*this->Nx()); 
        for(unsigned i=0; i<x_vec.size(); i++) x_vec[i] = this->abscissasX()[i];
        thrust::host_vector<double> y_vec(this->n()*this->Ny());
        for(unsigned i=0; i<y_vec.size(); i++) y_vec[i] = this->abscissasY()[i*x_vec.size()];
        detail::FieldFinv fpsiMinv_(gp, 500);
        detail::construct_rz( fpsi, fpsiMinv_, gp, psi_0, psi_1, this->x0(), this->x1(), 
                x_vec, y_vec, r_, z_, yr_, yz_, xr_, xz_, f_x_, g_);
        construct_rz( ); //lift to 3D grid and multiply by refined weights
        construct_metric(gp);
    }

    /**
     * @brief 1D version of the f_1 vector
     *
     * @return 
     */
    const thrust::host_vector<double>& f1_x()const{return f_x_;}
    /**
     * @brief 2D version of the f_2 vector
     *
     * @return 
     */
    const thrust::host_vector<double>& f2_xy()const{return f2_xy_;}
    /**
     * @brief Get the whole f_1 vector
     *
     * @return 
     */
    const thrust::host_vector<double>& f1()const{return f_;}
    /**
     * @brief Get the whole f_2 vector
     *
     * @return 
     */
    const thrust::host_vector<double>& f2()const{return g_;}
    perpendicular_grid perp_grid() const { return orthogonal::refined::RingGrid2d<container>(*this);}
    const orthogonal::RingGrid3d<container>& associated() const{ return g_assoc_;}

    const thrust::host_vector<double>& r()const{return r_;}
    const thrust::host_vector<double>& z()const{return z_;}
    const thrust::host_vector<double>& xr()const{return xr_;}
    const thrust::host_vector<double>& yr()const{return yr_;}
    const thrust::host_vector<double>& xz()const{return xz_;}
    const thrust::host_vector<double>& yz()const{return yz_;}
    const container& g_xx()const{return g_xx_;}
    const container& g_yy()const{return g_yy_;}
    const container& g_xy()const{return g_xy_;}
    const container& g_pp()const{return g_pp_;}
    const container& vol()const{return vol_;}
    const container& perpVol()const{return vol2d_;}
    private:
    void construct_rz( )
    {
        f_.resize( size());
        r_.resize(size()), z_.resize(size()), g_.resize(size());
        xr_.resize(size()), xz_.resize(size()), yr_.resize(size());
        yz_.resize(size());
        //lift to 3D grid
        unsigned Nx = this->n()*this->Nx(), Ny = this->n()*this->Ny();
        for( unsigned i=0; i<Nx; i++)
            f_x_[i] = -1./f_x_[i];
        for( unsigned j=0; j<Ny; j++)
            for( unsigned i=0; i<Nx; i++)
                f_[j*Nx+i] = f_x_[i];
        f2_xy_.resize(Nx*Ny);
        for( unsigned i=0; i<Nx*Ny; i++)
            f2_xy_[i] = g_[i];
        thrust::host_vector<double> wx = this->weightsX();
        thrust::host_vector<double> wy = this->weightsY();
        for( unsigned k=1; k<this->Nz(); k++)
            for( unsigned i=0; i<Nx*Ny; i++)
            {
                f_[k*Nx*Ny+i] = f_[(k-1)*Nx*Ny+i];
                g_[k*Nx*Ny+i] = g_[(k-1)*Nx*Ny+i];
                r_[k*Nx*Ny+i] = r_[(k-1)*Nx*Ny+i];
                z_[k*Nx*Ny+i] = z_[(k-1)*Nx*Ny+i];
                yr_[k*Nx*Ny+i] = yr_[(k-1)*Nx*Ny+i]*wy[k*Nx*Ny+i];
                yz_[k*Nx*Ny+i] = yz_[(k-1)*Nx*Ny+i]*wy[k*Nx*Ny+i];
                xr_[k*Nx*Ny+i] = xr_[(k-1)*Nx*Ny+i]*wx[k*Nx*Ny+i];
                xz_[k*Nx*Ny+i] = xz_[(k-1)*Nx*Ny+i]*wx[k*Nx*Ny+i];
            }
    }
    //compute metric elements from xr, xz, yr, yz, r and z
    void construct_metric( const solovev::GeomParameters& gp)
    {
        solovev::PsipR psipR_(gp); solovev::PsipZ psipZ_(gp);
        thrust::host_vector<double> tempxx( r_), tempxy(r_), tempyy(r_), tempvol(r_);
        for( unsigned i = 0; i<this->size(); i++)
        {
            double psipR = psipR_(r_[i], z_[i]), psipZ = psipZ_( r_[i], z_[i]);
            tempxx[i] = (xr_[i]*xr_[i]+xz_[i]*xz_[i]);
            tempxy[i] = (yr_[i]*xr_[i]+yz_[i]*xz_[i]);
            tempyy[i] = (yr_[i]*yr_[i]+yz_[i]*yz_[i]);
            //tempvol[i] = r_[i]/(f_[i]*f_[i] + tempxx[i]);
            //tempvol[i] = r_[i]/sqrt( tempxx[i]*tempyy[i] - tempxy[i]*tempxy[i] );
            tempvol[i] = r_[i]/sqrt( tempxx[i]*tempyy[i] );
            //tempvol[i] = r_[i]/fabs(f_[i]*g_[i])/(psipR*psipR + psipZ*psipZ);
        }
        g_xx_=tempxx, g_xy_=tempxy, g_yy_=tempyy, vol_=tempvol;
        dg::blas1::pointwiseDivide( tempvol, r_, tempvol);
        vol2d_ = tempvol;
        thrust::host_vector<double> ones = dg::evaluate( dg::one, *this);
        dg::blas1::pointwiseDivide( ones, r_, tempxx);
        dg::blas1::pointwiseDivide( tempxx, r_, tempxx); //1/R^2
        g_pp_=tempxx;
    }
    thrust::host_vector<double> f_x_, f2_xy_, f_, g_; 
    thrust::host_vector<double> r_, z_, xr_, xz_, yr_, yz_; 
    container g_xx_, g_xy_, g_yy_, g_pp_, vol_, vol2d_;
    orthogonal::RingGrid3d<container> g_assoc_;
    
    //The following points might also be useful for external grid generation
    //thrust::host_vector<double> r_0y, r_1y, z_0y, z_1y; //boundary points in x
    //thrust::host_vector<double> r_x0, r_x1, z_x0, z_x1; //boundary points in y

};

/**
 * @brief A three-dimensional grid based on "almost-conformal" coordinates by Ribeiro and Scott 2010
 */
template< class container>
struct RingGrid2d : public dg::refined::Grid2d
{
    typedef dg::OrthogonalCylindricalTag metric_category;
    RingGrid2d( unsigned multiple_x, unsigned multiple_y, const solovev::GeomParameters gp, double psi_0, double psi_1, unsigned n, unsigned n_old, unsigned Nx, unsigned Ny, dg::bc bcx): 
        dg::refined::Grid2d( multiple_x, multiple_y, 0, 1., 0., 2*M_PI, n,n_old,Nx,Ny, bcx, dg::PER),
        g_assoc_( gp, psi_0, psi_1, n_old, Nx, Ny, bcx) 
    {
        orthogonal::refined::RingGrid3d<container> g( multiple_x, multiple_y, gp, psi_0, psi_1, n,Nx,Ny,1,bcx);
        init_X_boundaries( g.x0(), g.x1());
        f_x_ = g.f1_x(), f_ = g.f1(), g_ = g.f2();
        r_=g.r(), z_=g.z(), xr_=g.xr(), xz_=g.xz(), yr_=g.yr(), yz_=g.yz();
        g_xx_=g.g_xx(), g_xy_=g.g_xy(), g_yy_=g.g_yy();
        vol2d_=g.perpVol();
    }
    RingGrid2d( const RingGrid3d<container>& g):
        dg::refined::Grid2d( g ), g_assoc_(g.associated())
    {
        f_x_ = g.f1_x();
        unsigned s = this->size();
        f_.resize(s), g_.resize(s), r_.resize( s), z_.resize(s), xr_.resize(s), xz_.resize(s), yr_.resize(s), yz_.resize(s);
        g_xx_.resize( s), g_xy_.resize(s), g_yy_.resize(s), vol2d_.resize(s);
        for( unsigned i=0; i<s; i++)
        {f_[i] = g.f1()[i], g_[i] = g.f2()[i], r_[i]=g.r()[i], z_[i]=g.z()[i], xr_[i]=g.xr()[i], xz_[i]=g.xz()[i], yr_[i]=g.yr()[i], yz_[i]=g.yz()[i];}
        thrust::copy( g.g_xx().begin(), g.g_xx().begin()+s, g_xx_.begin());
        thrust::copy( g.g_xy().begin(), g.g_xy().begin()+s, g_xy_.begin());
        thrust::copy( g.g_yy().begin(), g.g_yy().begin()+s, g_yy_.begin());
        thrust::copy( g.perpVol().begin(), g.perpVol().begin()+s, vol2d_.begin());
    }

    const orthogonal::RingGrid2d<container>& associated()const{return g_assoc_;}
    const thrust::host_vector<double>& f1_x()const{return f_x_;}
    const thrust::host_vector<double>& f2_xy()const{return g_;}
    const thrust::host_vector<double>& f1()const{return f_;}
    const thrust::host_vector<double>& f2()const{return g_;}

    const thrust::host_vector<double>& r()const{return r_;}
    const thrust::host_vector<double>& z()const{return z_;}
    const thrust::host_vector<double>& xr()const{return xr_;}
    const thrust::host_vector<double>& yr()const{return yr_;}
    const thrust::host_vector<double>& xz()const{return xz_;}
    const thrust::host_vector<double>& yz()const{return yz_;}
    const container& g_xx()const{return g_xx_;}
    const container& g_yy()const{return g_yy_;}
    const container& g_xy()const{return g_xy_;}
    const container& vol()const{return vol2d_;}
    const container& perpVol()const{return vol2d_;}
    private:
    thrust::host_vector<double> f_x_, f_, g_; //1d & 2d vector
    thrust::host_vector<double> r_, z_, xr_, xz_, yr_, yz_; //2d vector
    container g_xx_, g_xy_, g_yy_, vol2d_;
    orthogonal::RingGrid2d<container> g_assoc_;
};

/**
 * @brief Integrates the equations for a field line and 1/B
 */ 
}//namespace refined
}//namespace orthogonal
namespace dg{
/**
 * @brief This function pulls back a function defined in cartesian coordinates R,Z to the conformal coordinates x,y,\phi
 *
 * i.e. F(x,y) = f(R(x,y), Z(x,y))
 * @tparam BinaryOp The function object 
 * @param f The function defined on R,Z
 * @param g The grid
 *
 * @return A set of points representing F(x,y)
 */
template< class BinaryOp, class container>
thrust::host_vector<double> pullback( BinaryOp f, const orthogonal::refined::RingGrid2d<container>& g)
{
    thrust::host_vector<double> vec( g.size());
    for( unsigned i=0; i<g.size(); i++)
        vec[i] = f( g.r()[i], g.z()[i]);
    return vec;
}
///@cond
template<class container>
thrust::host_vector<double> pullback( double(f)(double,double), const orthogonal::refined::RingGrid2d<container>& g)
{
    return pullback<double(double,double),container>( f, g);
}
///@endcond
/**
 * @brief This function pulls back a function defined in cylindrical coordinates R,Z,\phi to the conformal coordinates x,y,\phi
 *
 * i.e. F(x,y,\phi) = f(R(x,y), Z(x,y), \phi)
 * @tparam TernaryOp The function object 
 * @param f The function defined on R,Z,\phi
 * @param g The grid
 *
 * @return A set of points representing F(x,y,\phi)
 */
template< class TernaryOp, class container>
thrust::host_vector<double> pullback( TernaryOp f, const orthogonal::refined::RingGrid3d<container>& g)
{
    thrust::host_vector<double> vec( g.size());
    unsigned size2d = g.n()*g.n()*g.Nx()*g.Ny();
    Grid1d<double> gz( g.z0(), g.z1(), 1, g.Nz());
    thrust::host_vector<double> absz = create::abscissas( gz);
    for( unsigned k=0; k<g.Nz(); k++)
        for( unsigned i=0; i<size2d; i++)
            vec[k*size2d+i] = f( g.r()[k*size2d+i], g.z()[k*size2d+i], absz[k]);
            //vec[k*size2d+i] = f( g.r()[i], g.z()[i], absz[k]);
    return vec;
}
///@cond
template<class container>
thrust::host_vector<double> pullback( double(f)(double,double,double), const orthogonal::refined::RingGrid3d<container>& g)
{
    return pullback<double(double,double,double),container>( f, g);
}
///@endcond

}//namespace dg
