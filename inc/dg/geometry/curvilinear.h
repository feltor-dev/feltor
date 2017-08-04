#pragma once

#include "dg/backend/manage.h"
#include "dg/blas1.h"
#include "geometry.h"
#include "geometry_traits.h"
#include "generator.h"

namespace dg
{
///@addtogroup basicgrids
///@{

///@cond
template< class container>
struct CurvilinearGrid2d; 
///@endcond

//when we make a 3d grid with eta and phi swapped the metric structure and the transformation changes 
//In practise it can only be orthogonal due to the projection matrix in the elliptic operator


/**
 * @brief A three-dimensional grid based on curvilinear coordinates
 @tparam container models aContainer
 */
template< class container>
struct CurvilinearGrid3d : public dg::aGeometry3d
{
    typedef CurvilinearGrid2d<container> perpendicular_grid;

    CurvilinearGrid3d( double R0, double R1, double Z0, double Z1, double phi0, double phi1, unsigned n, unsigned NR, unsigned NZ, unsigned Nphi, bc bcr = PER, bc bcz = PER, bc bcphi = PER): 
        dg::Grid3d(0,R1-R0,0,Z1-Z0,phi0,phi1,n,NR,NZ,Nphi,bcR,bcZ,bcphi)
        {
            handle_.set( new ShiftedIdentityGenerator(R0,R1,Z0,Z1));
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
    CurvilinearGrid3d( const geo::aGenerator& generator, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx=dg::DIR):
        dg::Grid3d( 0, generator.width(), 0., generator.height(), 0., 2.*M_PI, n, Nx, Ny, Nz, bcx, dg::PER, dg::PER)
    { 
        handle_ = generator;
        construct( n, Nx, Ny,Nz);
    }

    perpendicular_grid perp_grid() const { return perpendicular_grid(*this);}

    /**
    * @brief Function to do the pullback
    * @note Don't call! Use the pullback() function
    */
    template<class TernaryOp>
    thrust::host_vector<double> doPullback( TernaryOp f)const {
        thrust::host_vector<double> vec( size());
        unsigned size2d = n()*n()*Nx()*Ny();
        for( unsigned k=0; k<Nz(); k++)
            for( unsigned i=0; i<size2d; i++)
                vec[k*size2d+i] = f( r_[i], z_[i], phi_[k]);
        return vec;
    }
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
    const geo::aGenerator & generator() const{return handle_.get()}
    bool isOrthonormal() const { return handle_.get().isOrthonormal();}
    bool isOrthogonal() const { return handle_.get().isOrthogonal();}
    bool isConformal() const { return handle_.get().isConformal();}
    private:
    virtual void do_set( unsigned new_n, unsigned new_Nx, unsigned new_Ny,unsigned new_Nz){
        dg::Grid3d::do_set( new_n, new_Nx, new_Ny,new_Nz);
        construct( new_n, new_Nx, new_Ny,new_Nz);
    }
    void construct( unsigned n, unsigned Nx, unsigned Ny,unsigned Nz)
    {
        dg::Grid1d gY1d( 0., y0(), n, Ny, dg::PER);
        dg::Grid1d gX1d( 0., x0(), n, Nx);
        dg::Grid1d gphi( z0(), z1(), 1, Nz);
        phi_ = dg::create::abscissas( gphi);
        thrust::host_vector<double> x_vec = dg::evaluate( dg::cooX1d, gX1d);
        thrust::host_vector<double> y_vec = dg::evaluate( dg::cooX1d, gY1d);
        handle_.get()( x_vec, y_vec, r_, z_, xr_, xz_, yr_, yz_);
        //lift to 3D grid
        unsigned size = this->size();
        xr_.resize(size), yr_.resize( size), xz_.resize( size), yz_.resize(size);
        unsigned size2d = this->n()*this->n()*this->Nx()*this->Ny();
        for( unsigned k=1; k<this->Nz(); k++)
            for( unsigned i=0; i<size2d; i++)
            {
                xr_[k*size2d+i] = xr_[(k-1)*size2d+i];
                xz_[k*size2d+i] = xz_[(k-1)*size2d+i];
                yr_[k*size2d+i] = yr_[(k-1)*size2d+i];
                yz_[k*size2d+i] = yz_[(k-1)*size2d+i];
            }
        construct_metric();
    }
    //compute metric elements from xr, xz, yr, yz, r and z
    void construct_metric( )
    {
        thrust::host_vector<double> tempxx( size()), tempxy(size()), tempyy(size()), tempvol(size()), tempvol2d(size()), temppp(size());
        unsigned size2d = this->n()*this->n()*this->Nx()*this->Ny();
        for( unsigned i = 0; i<this->Nz(); i++)
        for( unsigned j = 0; j<size2d; j++)
        {
            unsigned idx = i*size2d+j;
            tempxx[idx] = (xr_[idx]*xr_[idx]+xz_[idx]*xz_[idx]);
            tempxy[idx] = (yr_[idx]*xr_[idx]+yz_[idx]*xz_[idx]);
            tempyy[idx] = (yr_[idx]*yr_[idx]+yz_[idx]*yz_[idx]);
            tempvol2d[idx] = 1./sqrt( tempxx[idx]*tempyy[idx] -tempxy[idx]*tempxy[idx] );
            tempvol[idx] = r_[j]*tempvol2d[idx];
            temppp[idx] = 1./r_[j]/r_[j];
        }
        //now transfer to device
        dg::blas1::transfer(tempxx,     g_xx_);
        dg::blas1::transfer(tempxy,     g_xy_);
        dg::blas1::transfer(tempyy,     g_yy_);
        dg::blas1::transfer(tempvol,    vol_);
        dg::blas1::transfer(tempvol2d,  vol2d_);
        dg::blas1::transfer(temppp,     g_pp_);
    }
    thrust::host_vector<double> r_, z_, phi_;  //2d and 1d vectors
    thrust::host_vector<double> xr_, xz_, yr_, yz_;
    container g_xx_, g_xy_, g_yy_, g_pp_, vol_, vol2d_;
    dg::Handle<geo::aGenerator> handle_;
};

/**
 * @brief A three-dimensional grid based on curvilinear coordinates
 */
template< class container>
struct CurvilinearGrid2d : public dg::aGeometry2d
{
    CurvilinearGrid2d( double R0, double R1, double Z0, double Z1, unsigned n, unsigned NR, unsigned NZ, unsigned Nphi, bc bcR = PER, bc bcZ = PER): 
        dg::Grid2d(0,R1-R0,0,Z1-Z0,n,NR,NZ,bcR,bcZ), handle_(new ShiftedIdentityGenerator(R0,R1,Z0,Z1))
        {
            construct( n, NR, NZ);
        }
    /*!@brief Constructor
    
     * @param generator must generate an orthogonal grid (class takes ownership of the pointer)
     * @param n number of polynomial coefficients
     * @param Nx number of cells in first coordinate
     * @param Ny number of cells in second coordinate
     * @param bcx boundary condition in first coordinate
     */
    CurvilinearGrid2d( const geo::aGenerator& generator, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx=dg::DIR):
        dg::Grid2d( 0, generator.width(), 0., generator.height(), n, Nx, Ny, bcx, dg::PER), handle_(generator)
    {
        construct( n,Nx,Ny);
    }
    CurvilinearGrid2d( const CurvilinearGrid3d<container>& g):
        dg::Grid2d( g.x0(), g.x1(), g.y0(), g.y1(), g.n(), g.Nx(), g.Ny(), g.bcx(), g.bcy()), handle_(g.generator())
    {
        r_ = g.r();
        z_ = g.z();
        unsigned s = this->size();
        xr_.resize(s), xz_.resize(s), yr_.resize(s), yz_.resize(s);
        g_xx_.resize( s), g_xy_.resize(s), g_yy_.resize(s), vol2d_.resize(s);
        for( unsigned i=0; i<s; i++) { 
            xr_[i]=g.xr()[i], xz_[i]=g.xz()[i], yr_[i]=g.yr()[i], yz_[i]=g.yz()[i];
        }
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
    const geo::aGenerator& generator() const{return handle_.get();}
    bool isOrthonormal() const { return generator_->isOrthonormal();}
    bool isOrthogonal() const { return generator_->isOrthogonal();}
    bool isConformal() const { return generator_->isConformal();}
    private:
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny)
    {
        dg::Grid2d::do_set( new_n, new_Nx, new_Ny);
        construct( new_n, new_Nx, new_Ny);
    }
    void construct( unsigned n, unsigned Nx, unsigned Ny)
    {
        CurvilinearGrid3d<container> g( handle_, n,Nx,Ny,1,bcx());
        r_=g.r(), z_=g.z(), xr_=g.xr(), xz_=g.xz(), yr_=g.yr(), yz_=g.yz();
        g_xx_=g.g_xx(), g_xy_=g.g_xy(), g_yy_=g.g_yy();
        vol2d_=g.perpVol();
    }
    thrust::host_vector<double> r_, z_, xr_, xz_, yr_, yz_;
    container g_xx_, g_xy_, g_yy_, vol2d_;
    dg::Handle<geo::aGenerator> handle_;
};

///@}
}//namespace dg
