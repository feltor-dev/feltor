#pragma once

#include "dg/backend/grid.h"
#include "dg/blas1.h"
#include "dg/geometry/geometry_traits.h"
#include "generator.h"

namespace dg
{
///@addtogroup grids
///@{

///@cond
template< class container>
struct CurvilinearGrid2d; 
///@endcond

/**
 * @brief A three-dimensional grid based on curvilinear coordinates
 @tparam container models aContainer
 */
template< class container>
struct CylindricalGrid3d : public dg::Grid3d
{
    typedef dg::CurvilinearPerpTag metric_category;
    typedef CurvilinearGrid2d<container> perpendicular_grid;

    CylindricalGrid3d( double R0, double R1, double Z0, double Z1, double phi0, double phi1, unsigned n, unsigned NR, unsigned NZ, unsigned Nphi, bc bcr = PER, bc bcz = PER, bc bcphi = PER): 
        dg::Grid3d(0,R1-R0,0,Z1-Z0,phi0,phi1,n,NR,NZ,Nphi,bcR,bcZ,bcphi)
        {
            generator_ = new IdentityGenerator(R0,R1,Z0,Z1);
            construct( n, NR, NZ,Nphi);
        }

    /*!@brief Constructor
    
     * the coordinates of the computational space are called x,y,z
     * @param generator must generate a grid
     * @param n number of %Gaussian nodes in x and y
     * @param Nx number of cells in x
     @param Ny number of cells in y 
     @param Nz  number of cells z
     @param bcx boundary condition in x
     @note the boundary conditions for y and z are set periodic
     */
    CylindricalGrid3d( const geo::aGenerator* generator, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx=dg::DIR):
        dg::Grid3d( 0, generator->width(), 0., generator->height(), 0., 2.*M_PI, n, Nx, Ny, Nz, bcx, dg::PER, dg::PER)
    { 
        generator_ = generator;
        construct( n, Nx, Ny,Nz);
    }

    perpendicular_grid perp_grid() const { return perpendicular_grid(*this);}
    /// a 2d vector containing R(x,y)
    const thrust::host_vector<double>& r()const{return r_;}
    /// a 2d vector containing Z(x,y)
    const thrust::host_vector<double>& z()const{return z_;}
    /// a 1d vector containing the phi values
    const thrust::host_vector<double>& phi()const{return phi_;}
    /// a 3d vector containing dx/dR
    const thrust::host_vector<double>& xr()const{return xr_;}
    /// a 3d vector containing dy/dR
    const thrust::host_vector<double>& yr()const{return yr_;}
    /// a 3d vector containing dx/dZ
    const thrust::host_vector<double>& xz()const{return xz_;}
    /// a 3d vector containing dy/dZ
    const thrust::host_vector<double>& yz()const{return yz_;}
    const container& g_xx()const{return g_xx_;}
    const container& g_yy()const{return g_yy_;}
    const container& g_xy()const{return g_xy_;}
    const container& g_pp()const{return g_pp_;}
    const container& vol()const{return vol_;}
    const container& perpVol()const{return vol2d_;}
    const geo::aGenerator & generator() const{return *generator_;}
    bool isOrthonormal() const { return generator_->isOrthonormal();}
    bool isOrthogonal() const { return generator_->isOrthogonal();}
    bool isConformal() const { return generator_->isConformal();}
    CylindricalGrid3d( const CylindricalGrid3d& src):Grid3d(src), g_xx_(src.g_xx_),g_xy_(src.g_xy_),g_yy_(src.g_yy_),g_pp_(src.g_pp_),vol_(src.vol_),vol2d_(src.vol2d_)
    {
        r_=src.r_,z_=src.z_,phi_=src.phi_,xr_=src.xr_,xz_=src.xz_,yr_=src.yr_,yz_=src.yz_;
        generator_ = src.generator_->clone();
    }
    CylindricalGrid3d& operator=( const CylindricalGrid3d& src)
    {
        Grid3d::operator=(src);//call base class assignment
        if( &src!=this)
        {
            delete generator_;
            g_xx_=src.g_xx_,g_xy_=src.g_xy_,g_yy_=src.g_yy_,g_pp_=src.g_pp_,vol_=src.vol_,vol2d_=src.vol2d_;
            r_=src.r_,z_=src.z_,phi_=src.phi_,xr_=src.xr_,xz_=src.xz_,yr_=src.yr_,yz_=src.yz_;
            generator_ = src.generator_->clone();
        }
        return *this;
    }
    virtual ~CylindricalGrid3d(){
        delete generator_;
    }
    private:
    virtual void do_set( unsigned new_n, unsigned new_Nx, unsigned new_Ny,unsigned new_Nz){
        dg::Grid3d::do_set( new_n, new_Nx, new_Ny,new_Nz);
        construct( new_n, new_Nx, new_Ny,new_Nz);
    }
    void construct( unsigned n, unsigned Nx, unsigned Ny,unsigned Nz)
    {
        dg::Grid1d gY1d( 0, generator_->height(), n, Ny, dg::PER);
        dg::Grid1d gX1d( 0., generator_->width(), n, Nx);
        dg::Grid1d gphi( z0(), z1(), 1, Nz);
        phi_ = dg::create::abscissas( gphi);
        thrust::host_vector<double> x_vec = dg::evaluate( dg::cooX1d, gX1d);
        thrust::host_vector<double> y_vec = dg::evaluate( dg::cooX1d, gY1d);
        (*generator_)( x_vec, y_vec, r_, z_, xr_, xz_, yr_, yz_);
        init_X_boundaries( 0., generator_->width());
        lift3d( ); //lift to 3D grid
        construct_metric();
    }
    void lift3d( )
    {
        //lift to 3D grid
        unsigned size = this->size();
        xr_.resize(size), yr_.resize( size), xz_.resize( size), yz_.resize(size);
        unsigned Nx = this->n()*this->Nx(), Ny = this->n()*this->Ny();
        for( unsigned k=1; k<this->Nz(); k++)
            for( unsigned i=0; i<Nx*Ny; i++)
            {
                xr_[k*Nx*Ny+i] = xr_[(k-1)*Nx*Ny+i];
                xz_[k*Nx*Ny+i] = xz_[(k-1)*Nx*Ny+i];
                yr_[k*Nx*Ny+i] = yr_[(k-1)*Nx*Ny+i];
                yz_[k*Nx*Ny+i] = yz_[(k-1)*Nx*Ny+i];
            }
    }
    //compute metric elements from xr, xz, yr, yz, r and z
    void construct_metric( )
    {
        thrust::host_vector<double> tempxx( xr_), tempxy(xr_), tempyy(xr_), tempvol(xr_), tempvol2d(xr_);
        unsigned size2d = this->n()*this->n()*this->Nx()*this->Ny();
        for( unsigned i = 0; i<this->Nz(); i++)
        for( unsigned j = 0; j<size2d; j++)
        {
            unsigned idx = i*size2d+j;
            tempxx[idx] = (xr_[idx]*xr_[idx]+xz_[idx]*xz_[idx]);
            tempxy[idx] = (yr_[idx]*xr_[idx]+yz_[idx]*xz_[idx]);
            tempyy[idx] = (yr_[idx]*yr_[idx]+yz_[idx]*yz_[idx]);
            tempvol[idx] = r_[j]/sqrt( tempxx[idx]*tempyy[idx] -tempxy[idx]*tempxy[idx] );
        }
        g_xx_=tempxx, g_xy_=tempxy, g_yy_=tempyy, vol_=tempvol;
        dg::blas1::pointwiseDivide( tempvol, r_, tempvol);
        vol2d_ = tempvol;
        thrust::host_vector<double> ones = dg::evaluate( dg::one, *this);
        dg::blas1::pointwiseDivide( ones, r_, tempxx);
        dg::blas1::pointwiseDivide( tempxx, r_, tempxx); //1/R^2
        g_pp_=tempxx;
    }
    thrust::host_vector<double> r_, z_, phi_, xr_, xz_, yr_, yz_;
    container g_xx_, g_xy_, g_yy_, g_pp_, vol_, vol2d_;
    geo::aGenerator* generator_;
};

/**
 * @brief A three-dimensional grid based on curvilinear coordinates
 */
template< class container>
struct CurvilinearGrid2d : public dg::Grid2d
{
    typedef dg::CurvilinearPerpTag metric_category;

    CurvilinearGrid2d( double R0, double R1, double Z0, double Z1, unsigned n, unsigned NR, unsigned NZ, unsigned Nphi, bc bcR = PER, bc bcZ = PER): 
        dg::Grid2d(0,R1-R0,0,Z1-Z0,n,NR,NZ,bcR,bcZ)
        {
            generator_ = new IdentityGenerator(R0,R1,Z0,Z1);
            construct( n, NR, NZ);
        }
    /*!@brief Constructor
    
     * @param generator must generate an orthogonal grid
     * @param n number of polynomial coefficients
     * @param Nx number of cells in first coordinate
     @param Ny number of cells in second coordinate
     @param bcx boundary condition in first coordinate
     */
    CurvilinearGrid2d( const geo::aGenerator* generator, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx=dg::DIR):
        dg::Grid2d( 0, generator->width(), 0., generator->height(), n, Nx, Ny, bcx, dg::PER)
    {
        construct( n,Nx,Ny);
    }
    CurvilinearGrid2d( const CylindricalGrid3d<container>& g):
        dg::Grid2d( g.x0(), g.x1(), g.y0(), g.y1(), g.n(), g.Nx(), g.Ny(), g.bcx(), g.bcy())
    {
        generator_ = g.generator();
        unsigned s = this->size();
        r_.resize( s), z_.resize(s), xr_.resize(s), xz_.resize(s), yr_.resize(s), yz_.resize(s);
        g_xx_.resize( s), g_xy_.resize(s), g_yy_.resize(s), vol2d_.resize(s);
        for( unsigned i=0; i<s; i++)
        { r_[i]=g.r()[i], z_[i]=g.z()[i], xr_[i]=g.xr()[i], xz_[i]=g.xz()[i], yr_[i]=g.yr()[i], yz_[i]=g.yz()[i];}
        thrust::copy( g.g_xx().begin(), g.g_xx().begin()+s, g_xx_.begin());
        thrust::copy( g.g_xy().begin(), g.g_xy().begin()+s, g_xy_.begin());
        thrust::copy( g.g_yy().begin(), g.g_yy().begin()+s, g_yy_.begin());
        thrust::copy( g.perpVol().begin(), g.perpVol().begin()+s, vol2d_.begin());
    }

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
    const geo::aGenerator& generator() const{return *generator_;}
    bool isOrthonormal() const { return generator_->isOrthonormal();}
    bool isOrthogonal() const { return generator_->isOrthogonal();}
    bool isConformal() const { return generator_->isConformal();}

    Curvilinear2d( const Curvilinear2d& src):Grid2d(src), g_xx_(src.g_xx_),g_xy_(src.g_xy_),g_yy_(src.g_yy_),vol2d_(src.vol2d_)
    {
        r_=src.r_,z_=src.z_,xr_=src.xr_,xz_=src.xz_,yr_=src.yr_,yz_=src.yz_;
        generator_ = src.generator_->clone();
    }
    Curvilinear2d& operator=( const Curvilinear2d& src)
    {
        Grid2d::operator=(src); //call base class assignment
        if( &src!=this)
        {
            delete generator_;
            g_xx_=src.g_xx_,g_xy_=src.g_xy_,g_yy_=src.g_yy_,vol2d_=src.vol2d_;
            r_=src.r_,z_=src.z_,xr_=src.xr_,xz_=src.xz_,yr_=src.yr_,yz_=src.yz_;
            generator_ = src.generator_->clone();
        }
        return *this;
    }
    virtual ~Curvilinear2d(){
        delete generator_;
    }
    private:
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny)
    {
        dg::Grid2d::do_set( new_n, new_Nx, new_Ny);
        construct( new_n, new_Nx, new_Ny);
    }
    void construct( unsigned n, unsigned Nx, unsigned Ny)
    {
        CylindricalGrid3d<container> g( generator_, n,Nx,Ny,1,bcx());
        r_=g.r(), z_=g.z(), xr_=g.xr(), xz_=g.xz(), yr_=g.yr(), yz_=g.yz();
        g_xx_=g.g_xx(), g_xy_=g.g_xy(), g_yy_=g.g_yy();
        vol2d_=g.perpVol();
    }
    thrust::host_vector<double> r_, z_, xr_, xz_, yr_, yz_;
    container g_xx_, g_xy_, g_yy_, vol2d_;
    geo::aGenerator* generator_;
};

///@}
}//namespace dg
