#pragma once

#include <mpi.h>

#include "dg/backend/mpi_evaluation.h"
#include "dg/backend/mpi_grid.h"
#include "dg/backend/mpi_vector.h"
#include "curvilinear.h"
#include "generator.h"



namespace dg
{

///@cond
struct CurvilinearMPIGrid2d; 
///@endcond
//
///@addtogroup grids
///@{

/**
 * This is s 2x1 product space grid
 * @tparam MPIContainer Vector class that holds metric coefficients
 */
struct CylindricalProductMPIGrid3d : public dg::aMPIGeometry3d
{
    typedef dg::CurvilinearMPIGrid2d<LocalContainer> perpendicular_grid; //!< the two-dimensional grid
    typedef typename MPIContainer::container_type LocalContainer; //!< the local container type
    /**
     * @copydoc Grid3d::Grid3d()
     * @param comm a three-dimensional Cartesian communicator
     * @note the paramateres given in the constructor are global parameters 
     */
    CylindricalMPIGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, MPI_Comm comm): 
        dg::MPIGrid3d( 0, x1-x0, 0, y1-y0, z0, z1, n, Nx, Ny, Nz, comm),
        handle_( new ShiftedIdentityGenerator(x0,x1,y0,y1), n,Nx,Ny,local().Nz(), bcx,bcy,bcz)
     {
        CylindricalGrid3d<LocalContainer> g(handle_.get(),n,Nx,Ny,this->Nz());
        divide_and_conquer();
     }

    /**
     * @copydoc Grid3d::Grid3d()
     * @param comm a three-dimensional Cartesian communicator
     * @note the paramateres given in the constructor are global parameters 
     */
    CylindricalMPIGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):
        dg::MPIGrid3d( 0, x1-x0, 0, y1-y0, z0, z1, n, Nx, Ny, Nz, bcx, bcy, bcz, comm),
        handle_( new ShiftedIdentityGenerator(x0,x1,y0,y1))
     {
        CylindricalGrid3d<LocalContainer> g(handle_.get(),n,Nx,Ny,this->Nz());
        divide_and_conquer(g);
     }

    CylindricalMPIGrid3d( const geo::aGenerator& generator, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx, MPI_Comm comm): 
        dg::MPIGrid3d( 0, generator.width(), 0., generator.height(), 0., 2.*M_PI, n, Nx, Ny, Nz, bcx, dg::PER, dg::PER, comm),
        handle_( generator)
    {
        CylindricalGrid3d<LocalContainer> g(generator,n,Nx,Ny,this->Nz());
        divide_and_conquer(g);
    }

    perpendicular_grid perp_grid() const { return perpendicular_grid(*this);}

    template<class TernaryOp>
    dg::MPI_Vector<thrust::host_vector<double> > doPullback(TernaryOp f)const{
        thrust::host_vector<double> vec( g.size());
        unsigned size2d = g.n()*g.n()*g.Nx()*g.Ny();
        for( unsigned k=0; k<g.Nz(); k++)
            for( unsigned i=0; i<size2d; i++)
                vec[k*size2d+i] = f( g.r()[i], g.z()[i], g.phi()[k]);
        MPI_Vector<thrust::host_vector<double> > v( vec, g.communicator());
        return v;
    }
    const dg::MPI_Vector<thrust::host_vector<double> >& xr()const{return xr_;}
    const dg::MPI_Vector<thrust::host_vector<double> >& yr()const{return yr_;}
    const dg::MPI_Vector<thrust::host_vector<double> >& xz()const{return xz_;}
    const dg::MPI_Vector<thrust::host_vector<double> >& yz()const{return yz_;}
    const MPIContainer& g_xx()const{return g_xx_;}
    const MPIContainer& g_yy()const{return g_yy_;}
    const MPIContainer& g_xy()const{return g_xy_;}
    const MPIContainer& g_pp()const{return g_pp_;}
    const MPIContainer& vol()const{return vol_;}
    const MPIContainer& perpVol()const{return vol2d_;}
    const geo::aGenerator& generator() const{return g.generator();}
    bool isOrthonormal() const { return g.isOrthonormal();}
    bool isOrthogonal() const { return g.isOrthogonal();}
    bool isConformal() const { return g.isConformal();}
    private:
    virtual void do_set( unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz)
    {
        dg::MPIGrid3d::do_set(new_n, new_Nx, new_Ny, new_Nz);
        dg::CylindricalGrid3d<thrust::host_vector> g( handle_.get(), new_n, new_Nx,new_Ny,Nz());
        divide_and_conquer(g);//distribute to processes
    }
    void divide_and_conquer( const dg::CylindricalGrid3d<thrust::host_vector<double> & g )
    {
        r_.resize( this->n()*this->n()*this->Nx()*this->Ny());
        z_ = r_;
        phi_.resize( this->Nz());
        xr_=dg::evaluate( dg::one, *this), xz_=xr_, yr_=xr_, yz_=xr_;
        dg::blas1::transfer( xr_, g_xx_);
        g_xy_=g_xx_, g_yy_=g_xx_, g_pp_=g_xx_, vol_=g_xx_, vol2d_=g_xx_;
        thrust::host_vector<double> tempxx(size()), tempxy(size()),tempyy(size()),temppp(size()),tempvol2d(size()),tempvol(size());
        //divide and conquer
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        init_X_boundaries( g.x0(), g.x1());
        for( unsigned s=0; s<this->Nz(); s++)
            //for( unsigned py=0; py<dims[1]; py++)
                for( unsigned i=0; i<this->n()*this->Ny(); i++)
                    //for( unsigned px=0; px<dims[0]; px++)
                        for( unsigned j=0; j<this->n()*this->Nx(); j++)
                        {
                            unsigned idx2D1 = i*this->n()*this->Nx() + j;
                            unsigned idx2D2 = ((coords[1]*this->n()*this->Ny()+i)*dims[0] + coords[0])*this->n()*this->Nx() + j;
                            r_[idx2D1] = g.r()[idx2D2];
                            z_[idx2D1] = g.z()[idx2D2];
                            phi_[s] = g.phi()[coords[2]*this->Nz()+s];
                            unsigned idx1 = (s*this->n()*this->Ny()+i)*this->n()*this->Nx() + j;
                            unsigned idx2 = (((s*dims[1]+coords[1])*this->n()*this->Ny()+i)*dims[0] + coords[0])*this->n()*this->Nx() + j;
                            xr_.data()[idx1] = g.xr()[idx2];
                            xz_.data()[idx1] = g.xz()[idx2];
                            yr_.data()[idx1] = g.yr()[idx2];
                            yz_.data()[idx1] = g.yz()[idx2];

                            tempxx[idx1] = g.g_xx()[idx2];
                            tempxy[idx1] = g.g_xy()[idx2];
                            tempyy[idx1] = g.g_yy()[idx2];
                            temppp[idx1] = g.g_pp()[idx2];
                            tempvol[idx1] = g.vol()[idx2];
                            tempvol2d[idx1] = g.perpVol()[idx2];
                        }
        dg::blas1::transfer( tempxx, g_xx_.data());
        dg::blas1::transfer( tempxy, g_xy_.data());
        dg::blas1::transfer( tempyy, g_yy_.data());
        dg::blas1::transfer( temppp, g_pp_.data());
        dg::blas1::transfer( tempvol, vol_.data());
        dg::blas1::transfer( tempvol2d, vol2d_.data());
    }
    thrust::host_vector<double> r_,z_,phi_;
    dg::MPI_Vector<thrust::host_vector<double> > xr_, xz_, yr_, yz_; //3d vector
    MPIContainer g_xx_, g_xy_, g_yy_, g_pp_, vol_, vol2d_;
    Handle<dg::geo::aGenerator> handle_;
};

/**
 * @tparam MPIContainer Vector class that holds metric coefficients
 */
struct CurvilinearMPIGrid2d : public dg::aMPIGeometry2d
{
    typedef typename MPIContainer::container_type LocalContainer; //!< the local container type

    CurvilinearMPIGrid2d( const geo::aGenerator& generator, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx, MPI_Comm comm2d): 
        dg::aMPIGeometry2d( 0, generator.width(), 0., generator.height(), n, Nx, Ny, bcx, dg::PER, comm2d),handle_(generator)
    {
        dg::CurvilinearGrid2d<thrust::host_vector<double> > g(generator, n, Nx, Ny);
        divide_and_conquer(g);
    }
    CurvilinearMPIGrid2d( const CylindricalMPIGrid3d<LocalContainer>& g):
        dg::aMPIGeometry2d( g.global().x0(), g.global().x1(), g.global().y0(), g.global().y1(), g.global().n(), g.global().Nx(), g.global().Ny(), g.global().bcx(), g.global().bcy(), get_reduced_comm( g.communicator() )),
        xr_(dg::evaluate( dg::one, *this)), xz_(xr_), yr_(xr_), yz_(xr_), 
        g_xx_(xr_), g_xy_(g_xx_), g_yy_(g_xx_), vol2d_(g_xx_),
        handle_(g.generator())
    {
        r_=g.r(); 
        z_=g.z(); 
        unsigned s = this->size();
        for( unsigned i=0; i<s; i++)
        {
            xr_.data()[i]=g.xr().data()[i]; 
            xz_.data()[i]=g.xz().data()[i]; 
            yr_.data()[i]=g.yr().data()[i]; 
            yz_.data()[i]=g.yz().data()[i];
        }
        thrust::copy( g.g_xx().data().begin(), g.g_xx().data().begin()+s, g_xx_.data().begin());
        thrust::copy( g.g_xy().data().begin(), g.g_xy().data().begin()+s, g_xy_.data().begin());
        thrust::copy( g.g_yy().data().begin(), g.g_yy().data().begin()+s, g_yy_.data().begin());
        thrust::copy( g.perpVol().data().begin(), g.perpVol().data().begin()+s, vol2d_.data().begin());
    }

    const thrust::host_vector<double>& r()const{return r_;}
    const thrust::host_vector<double>& z()const{return z_;}
    const dg::MPI_Vector<thrust::host_vector<double> >& xr()const{return xr_;}
    const dg::MPI_Vector<thrust::host_vector<double> >& yr()const{return yr_;}
    const dg::MPI_Vector<thrust::host_vector<double> >& xz()const{return xz_;}
    const dg::MPI_Vector<thrust::host_vector<double> >& yz()const{return yz_;}
    const MPIContainer& g_xx()const{return g_xx_;}
    const MPIContainer& g_yy()const{return g_yy_;}
    const MPIContainer& g_xy()const{return g_xy_;}
    const MPIContainer& vol()const{return vol2d_;}
    const geo::aGenerator& generator() const{return g.generator();}
    bool isOrthonormal() const { return g.isOrthonormal();}
    bool isOrthogonal() const { return g.isOrthogonal();}
    bool isConformal() const { return g.isConformal();}
    private:
    virtual void do_set( unsigned new_n, unsigned new_Nx, unsigned new_Ny)
    {
        dg::MPIGrid3d::do_set(new_n, new_Nx, new_Ny);
        dg::CurvilinearGrid2d<thrust::host_vector<double> > g( handle_.get(), new_n, new_Nx, new_Ny);
        divide_and_conquer(g);//distribute to processes
    }
    MPI_Comm get_reduced_comm( MPI_Comm src)
    {
        MPI_Comm planeComm;
        int remain_dims[] = {true,true,false}; //true true false
        MPI_Cart_sub( src, remain_dims, &planeComm);
        return planeComm;
    }
    void divide_and_conquer(const dg::CurvilinearGrid2d<thrust::host_vector<double> >& g_)
    {
        r_.resize( size());
        z_=r_;
        xr_=dg::evaluate( dg::one, *this), xz_=xr_, yr_=xr_, yz_=xr_;
        thrust::host_vector<double> tempxx(size()), tempxy(size()),tempyy(size()), tempvol(size());
        dg::blas1::transfer( xr_, g_xx_);
        g_xy_=g_xx_, g_yy_=g_xx_, vol2d_=g_xx_;
        //divide and conquer
        int dims[2], periods[2], coords[2];
        MPI_Cart_get( communicator(), 2, dims, periods, coords);
        //for( unsigned py=0; py<dims[1]; py++)
            for( unsigned i=0; i<this->n()*this->Ny(); i++)
                //for( unsigned px=0; px<dims[0]; px++)
                    for( unsigned j=0; j<this->n()*this->Nx(); j++)
                    {
                        unsigned idx1 = i*this->n()*this->Nx() + j;
                        unsigned idx2 = ((coords[1]*this->n()*this->Ny()+i)*dims[0] + coords[0])*this->n()*this->Nx() + j;
                        r_[idx1] = g_.r()[idx2];
                        z_[idx1] = g_.z()[idx2];
                        xr_.data()[idx1] = g_.xr()[idx2];
                        xz_.data()[idx1] = g_.xz()[idx2];
                        yr_.data()[idx1] = g_.yr()[idx2];
                        yz_.data()[idx1] = g_.yz()[idx2];
                        tempxx[idx1] = g_.g_xx()[idx2];
                        tempxy[idx1] = g_.g_xy()[idx2];
                        tempyy[idx1] = g_.g_yy()[idx2];
                        tempvol2d[idx1] = g_.perpVol()[idx2];
                    }
        dg::blas1::transfer( tempxx, g_xx_.data());
        dg::blas1::transfer( tempxy, g_xy_.data());
        dg::blas1::transfer( tempyy, g_yy_.data());
        dg::blas1::transfer( tempvol, vol2d_.data());
    }

    dg::SparseTensor<host_vector > jac_, metric_;
    std::vector<host_vector > map_;
    dg::Handle<dg::geo::aGenerator2d> handle_;
};
///@}
}//namespace dg

