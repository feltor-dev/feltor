#pragma once


#include "dg/geometry/refined_gridX.h"
#include "orthogonalX.h"



namespace orthogonal
{
namespace refined
{


template< class container>
struct GridX2d;

/**
 * @brief A three-dimensional grid based on "almost-conformal" coordinates by Ribeiro and Scott 2010
 */
template< class container>
struct GridX3d : public dg::refined::GridX3d
{
    typedef dg::CurvilinearCylindricalTag metric_category;
    typedef orthogonal::refined::GridX2d<container> perpendicular_grid;

    /**
     * @brief Construct 
     *
     * @param add_x
     * @param add_y
     * @param gp The geometric parameters define the magnetic field
     * @param psi_0 lower boundary for psi
     * @param fx factor in x-direction
     * @param fy factor in y-direction
     * @param n The dG number of polynomials
     * @param Nx The number of points in x-direction
     * @param Ny The number of points in y-direction
     * @param Nz The number of points in z-direction
     * @param bcx The boundary condition in x (z is periodic)
     * @param bcy The boundary condition in y (z is periodic)
     */
    GridX3d( unsigned add_x, unsigned add_y, unsigned howmanyX, unsigned howmanyY, solovev::GeomParameters gp, double psi_0, double fx, double fy, unsigned n, unsigned n_old, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx, dg::bc bcy): 
        dg::refined::GridX3d( add_x, add_y, howmanyX, howmanyY, 0,1, -2.*M_PI*fy/(1.-2.*fy), 2.*M_PI*(1.+fy/(1.-2.*fy)), 0., 2*M_PI, fx, fy, n, n_old, Nx, Ny, Nz, bcx, bcy, dg::PER),
        f_( this->size()), g_(f_), r_(f_), z_(f_), xr_(f_), xz_(f_), yr_(f_), yz_(f_),
        g_assoc_( gp, psi_0, fx, fy, n_old, Nx, Ny, Nz, bcx, bcy)
   // GridX3d( unsigned add_x, unsigned add_y, solovev::GeomParameters gp, double psi_0, double fx, double fy, unsigned n, unsigned n_old, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx, dg::bc bcy): 
   //     dg::refined::GridX3d( add_x, add_y, 0,1, -2.*M_PI*fy/(1.-2.*fy), 2.*M_PI*(1.+fy/(1.-2.*fy)), 0., 2*M_PI, fx, fy, n, n_old, Nx, Ny, Nz, bcx, bcy, dg::PER),
   //     f_( this->size()), g_(f_), r_(f_), z_(f_), xr_(f_), xz_(f_), yr_(f_), yz_(f_),
   //     g_assoc_( gp, psi_0, fx, fy, n_old, Nx, Ny, Nz, bcx, bcy)
    { 
        /////////////////////discretize x-direction and construct psi(x)
        assert( psi_0 < 0 );
        assert( gp.c[10] != 0);
        orthogonal::detail::FpsiX fpsi(gp);
        std::cout << "FIND X FOR PSI_0\n";
        const double x_0 = fpsi.find_x(psi_0);
        const double x_1 = -fx/(1.-fx)*x_0;
        std::cout << "X0 is "<<x_0<<" and X1 is "<<x_1<<"\n";
        init_X_boundaries( x_0, x_1);
        ////////////compute psi(x) for a grid on x 
        thrust::host_vector<double> x_vec(this->n()*this->Nx()), psi_x; 
        for(unsigned i=0; i<x_vec.size(); i++) {
            x_vec[i] = this->abscissasX()[i];
            //std::cout << x_vec[i]<<std::endl;
        }
        detail::XFieldFinv fpsiMinv_(gp, 500);
        psi_1_numerical_ = dg::detail::construct_psi_values( fpsiMinv_, gp, psi_0, this->x0(), x_vec, this->x1(), this->inner_Nx()*this->n(), psi_x, f_x_);
        /////////////////discretize y-direction and construct rest
        thrust::host_vector<double> rvec, zvec, yrvec, yzvec, gvec;
        thrust::host_vector<double> y_vec(this->n()*this->Ny());
        for(unsigned i=0; i<y_vec.size(); i++) y_vec[i] = this->abscissasY()[i*x_vec.size()];
        orthogonal::detail::construct_rz( fpsiMinv_, gp, 
                psi_0, psi_x, y_vec, 
                this->n()*this->outer_Ny(), 
                this->n()*(this->inner_Ny()+this->outer_Ny()), 
                rvec, zvec, yrvec, yzvec, gvec);
        unsigned Mx = this->n()*this->Nx(), My = this->n()*this->Ny();
        solovev::PsipR psipR_(gp);
        solovev::PsipZ psipZ_(gp);
        for( unsigned i=0; i<psi_x.size(); i++)
            for( unsigned j=0; j<y_vec.size(); j++)
            {
                xr_[j*Mx+i] = psipR_(rvec[j*Mx+i],zvec[j*Mx+i])/f_x_[i]; 
                xz_[j*Mx+i] = psipZ_(rvec[j*Mx+i],zvec[j*Mx+i])/f_x_[i]; 
                 f_[j*Mx + i] = -1./f_x_[i];
            }
        for( unsigned i=0; i<psi_x.size(); i++)
            f_x_[i] = -1./f_x_[i];
        //r_x1 = r_x0, z_x1 = z_x0; //periodic boundaries
        //now lift to 3D grid and multiply with refined weights
        thrust::host_vector<double> wx = this->weightsX();
        thrust::host_vector<double> wy = this->weightsY();
        for( unsigned k=0; k<this->Nz(); k++)
            for( unsigned i=0; i<Mx*My; i++)
            {
                f_[k*Mx*My+i] = f_[i]; g_[k*Mx*My+i] = gvec[i];
                r_[k*Mx*My+i] = rvec[i];
                z_[k*Mx*My+i] = zvec[i];
                yr_[k*Mx*My+i] = yrvec[i]*wy[k*Mx*My+i];
                yz_[k*Mx*My+i] = yzvec[i]*wy[k*Mx*My+i];
                xr_[k*Mx*My+i] = xr_[i]*wx[k*Mx*My+i];
                xz_[k*Mx*My+i] = xz_[i]*wx[k*Mx*My+i];
            }
        construct_metric();
    }


    const orthogonal::GridX3d<container>& associated() const{ return g_assoc_;}


    const thrust::host_vector<double>& f()const{return f_;}
    const thrust::host_vector<double>& g()const{return g_;}
    const thrust::host_vector<double>& r()const{return r_;}
    const thrust::host_vector<double>& z()const{return z_;}
    const thrust::host_vector<double>& xr()const{return xr_;}
    const thrust::host_vector<double>& yr()const{return yr_;}
    const thrust::host_vector<double>& xz()const{return xz_;}
    const thrust::host_vector<double>& yz()const{return yz_;}
    const thrust::host_vector<double>& f_x()const{return f_x_;}
    thrust::host_vector<double> x()const{
        dg::Grid1d<double> gx( x0(), x1(), n(), Nx());
        return dg::create::abscissas(gx);}
    const container& g_xx()const{return g_xx_;}
    const container& g_yy()const{return g_yy_;}
    const container& g_xy()const{return g_xy_;}
    const container& g_pp()const{return g_pp_;}
    const container& vol()const{return vol_;}
    const container& perpVol()const{return vol2d_;}
    perpendicular_grid perp_grid() const { return orthogonal::refined::GridX2d<container>(*this);}
    const thrust::host_vector<double>& rx0()const{return r_x0;}
    const thrust::host_vector<double>& zx0()const{return z_x0;}
    const thrust::host_vector<double>& rx1()const{return r_x1;}
    const thrust::host_vector<double>& zx1()const{return z_x1;}
    double psi1()const{return psi_1_numerical_;}
    private:
    //compute metric elements from xr, xz, yr, yz, r and z
    void construct_metric()
    {
        std::cout << "CONSTRUCTING METRIC\n";
        thrust::host_vector<double> tempxx( r_), tempxy(r_), tempyy(r_), tempvol(r_);
        unsigned Nx = this->n()*this->Nx(), Ny = this->n()*this->Ny();
        for( unsigned k=0; k<this->Nz(); k++)
            for( unsigned i=0; i<Ny; i++)
                for( unsigned j=0; j<Nx; j++)
                {
                    unsigned idx = k*Ny*Nx+i*Nx+j;
                    tempxx[idx] = (xr_[idx]*xr_[idx]+xz_[idx]*xz_[idx]);
                    tempxy[idx] = (yr_[idx]*xr_[idx]+yz_[idx]*xz_[idx]);
                    tempyy[idx] = (yr_[idx]*yr_[idx]+yz_[idx]*yz_[idx]);
                    tempvol[idx] = r_[idx]/sqrt(tempxx[idx]*tempyy[idx]-tempxy[idx]*tempxy[idx]);
                }
        g_xx_=tempxx, g_xy_=tempxy, g_yy_=tempyy, vol_=tempvol;
        dg::blas1::pointwiseDivide( tempvol, r_, tempvol);
        vol2d_ = tempvol;
        thrust::host_vector<double> ones = dg::evaluate( dg::one, *this);
        dg::blas1::pointwiseDivide( ones, r_, tempxx);
        dg::blas1::pointwiseDivide( tempxx, r_, tempxx); //1/R^2
        g_pp_=tempxx;
    }
    thrust::host_vector<double> f_x_; //1d vector
    thrust::host_vector<double> f_, g_, r_, z_, xr_, xz_, yr_, yz_; //3d vector
    container g_xx_, g_xy_, g_yy_, g_pp_, vol_, vol2d_;
    thrust::host_vector<double> r_x0, r_x1, z_x0, z_x1; //boundary points in y
    double psi_1_numerical_;
    orthogonal::GridX3d<container> g_assoc_;
};

/**
 * @brief A three-dimensional grid based on "almost-conformal" coordinates by Ribeiro and Scott 2010
 */
template< class container>
struct GridX2d : public dg::refined::GridX2d
{
    typedef dg::CurvilinearCylindricalTag metric_category;
    GridX2d( unsigned add_x, unsigned add_y, unsigned howmanyX, unsigned howmanyY, const solovev::GeomParameters gp, double psi_0, double fx, double fy, unsigned n, unsigned n_old, unsigned Nx, unsigned Ny, dg::bc bcx, dg::bc bcy): 
        dg::refined::GridX2d( add_x, add_y, howmanyX, howmanyY, 0, 1,-fy*2.*M_PI/(1.-2.*fy), 2*M_PI+fy*2.*M_PI/(1.-2.*fy), fx, fy, n, n_old, Nx, Ny, bcx, bcy),
    //GridX2d( unsigned add_x, unsigned add_y, const solovev::GeomParameters gp, double psi_0, double fx, double fy, unsigned n, unsigned n_old, unsigned Nx, unsigned Ny, dg::bc bcx, dg::bc bcy): 
    //    dg::refined::GridX2d( add_x, add_y, 0, 1,-fy*2.*M_PI/(1.-2.*fy), 2*M_PI+fy*2.*M_PI/(1.-2.*fy), fx, fy, n, n_old, Nx, Ny, bcx, bcy),
        g_assoc_( gp, psi_0, fx, fy, n_old, Nx, Ny, bcx, bcy) 
    {
        orthogonal::refined::GridX3d<container> g(add_x, add_y, howmanyX, howmanyY, gp, psi_0, fx,fy, n,Nx,Ny,1,bcx,bcy);
        init_X_boundaries( g.x0(), g.x1());
        //orthogonal::refined::GridX3d<container> g(add_x, add_y,  gp, psi_0, fx,fy, n,Nx,Ny,1,bcx,bcy);
        f_x_ = g.f_x();
        f_ = g.f(), g_ = g.g(), r_=g.r(), z_=g.z(), xr_=g.xr(), xz_=g.xz(), yr_=g.yr(), yz_=g.yz();
        g_xx_=g.g_xx(), g_xy_=g.g_xy(), g_yy_=g.g_yy();
        vol2d_=g.perpVol();
    }
    GridX2d( const GridX3d<container>& g):
        dg::refined::GridX2d(g), g_assoc_(g.associated())
    {
        f_x_ = g.f_x();
        unsigned s = this->size();
        f_.resize(s), g_.resize(s), r_.resize( s), z_.resize(s), xr_.resize(s), xz_.resize(s), yr_.resize(s), yz_.resize(s);
        g_xx_.resize( s), g_xy_.resize(s), g_yy_.resize(s), vol2d_.resize(s);
        for( unsigned i=0; i<s; i++)
        {f_[i] = g.f()[i], g_[i] = g.g()[i], r_[i]=g.r()[i], z_[i]=g.z()[i], xr_[i]=g.xr()[i], xz_[i]=g.xz()[i], yr_[i]=g.yr()[i], yz_[i]=g.yz()[i];}
        thrust::copy( g.g_xx().begin(), g.g_xx().begin()+s, g_xx_.begin());
        thrust::copy( g.g_xy().begin(), g.g_xy().begin()+s, g_xy_.begin());
        thrust::copy( g.g_yy().begin(), g.g_yy().begin()+s, g_yy_.begin());
        thrust::copy( g.perpVol().begin(), g.perpVol().begin()+s, vol2d_.begin());
    }
    const orthogonal::GridX2d<container>& associated()const{return g_assoc_;}
    const thrust::host_vector<double>& f()const{return f_;}
    const thrust::host_vector<double>& g()const{return g_;}
    const thrust::host_vector<double>& r()const{return r_;}
    const thrust::host_vector<double>& z()const{return z_;}
    const thrust::host_vector<double>& xr()const{return xr_;}
    const thrust::host_vector<double>& yr()const{return yr_;}
    const thrust::host_vector<double>& xz()const{return xz_;}
    const thrust::host_vector<double>& yz()const{return yz_;}
    thrust::host_vector<double> x()const{
        dg::Grid1d<double> gx( x0(), x1(), n(), Nx());
        return dg::create::abscissas(gx);}
    const thrust::host_vector<double>& f_x()const{return f_x_;}
    const container& g_xx()const{return g_xx_;}
    const container& g_yy()const{return g_yy_;}
    const container& g_xy()const{return g_xy_;}
    const container& vol()const{return vol2d_;}
    const container& perpVol()const{return vol2d_;}
    private:
    thrust::host_vector<double> f_x_; //1d vector
    thrust::host_vector<double> f_, g_, r_, z_, xr_, xz_, yr_, yz_; //2d vector
    container g_xx_, g_xy_, g_yy_, vol2d_;
    orthogonal::GridX2d<container> g_assoc_;
};

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
thrust::host_vector<double> pullback( BinaryOp f, const orthogonal::refined::GridX2d<container>& g)
{
    thrust::host_vector<double> vec( g.size());
    for( unsigned i=0; i<g.size(); i++)
        vec[i] = f( g.r()[i], g.z()[i]);
    return vec;
}

///@cond
template<class container>
thrust::host_vector<double> pullback( double(f)(double,double), const orthogonal::refined::GridX2d<container>& g)
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
thrust::host_vector<double> pullback( TernaryOp f, const orthogonal::refined::GridX3d<container>& g)
{
    thrust::host_vector<double> vec( g.size());
    unsigned size2d = g.n()*g.n()*g.Nx()*g.Ny();
    Grid1d<double> gz( g.z0(), g.z1(), 1, g.Nz());
    thrust::host_vector<double> absz = create::abscissas( gz);
    for( unsigned k=0; k<g.Nz(); k++)
        for( unsigned i=0; i<size2d; i++)
            vec[k*size2d+i] = f( g.r()[k*size2d+i], g.z()[k*size2d+i], absz[k]);
    return vec;
}
///@cond
template<class container>
thrust::host_vector<double> pullback( double(f)(double,double,double), const orthogonal::refined::GridX3d<container>& g)
{
    return pullback<double(double,double,double),container>( f, g);
}
///@endcond

}//namespace dg

