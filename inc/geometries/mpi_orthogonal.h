#pragma once

#include <mpi.h>

#include "dg/backend/mpi_grid.h"
#include "dg/backend/mpi_vector.h"
#include "orthogonal.h"



namespace dg
{

///@cond
template< class container>
struct OrthogonalMPIGrid2d; 
///@endcond
//
///@addtogroup grids
///@{

/**
 * @tparam LocalContainer Vector class that holds metric coefficients
 */
template<class LocalContainer>
struct OrthogonalMPIGrid3d : public dg::MPIGrid3d
{
    typedef dg::OrthogonalTag metric_category; //!< metric tag
    typedef dg::OrthogonalMPIGrid2d<LocalContainer> perpendicular_grid; //!< the two-dimensional grid

    template< class Generator>
    OrthogonalMPIGrid3d( const Generator& generator, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx, MPI_Comm comm): 
        dg::MPIGrid3d( 0, generator.width(), 0., generator.height(), 0., 2.*M_PI, n, Nx, Ny, Nz, bcx, dg::PER, dg::PER, comm),
        r_(dg::evaluate( dg::one, *this)), z_(r_), xr_(r_), xz_(r_), yr_(r_), yz_(r_),
        g_xx_(r_), g_xy_(g_xx_), g_yy_(g_xx_), g_pp_(g_xx_), vol_(g_xx_), vol2d_(g_xx_)
    {
        dg::OrthogonalGrid3d<LocalContainer> g( generator, n,Nx, Ny, local().Nz(), bcx);

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
                            unsigned idx1 = (s*this->n()*this->Ny()+i)*this->n()*this->Nx() + j;
                            unsigned idx2 = (((s*dims[1]+coords[1])*this->n()*this->Ny()+i)*dims[0] + coords[0])*this->n()*this->Nx() + j;
                            r_.data()[idx1] = g.r()[idx2];
                            z_.data()[idx1] = g.z()[idx2];
                            xr_.data()[idx1] = g.xr()[idx2];
                            xz_.data()[idx1] = g.xz()[idx2];
                            yr_.data()[idx1] = g.yr()[idx2];
                            yz_.data()[idx1] = g.yz()[idx2];
                            g_xx_.data()[idx1] = g.g_xx()[idx2];
                            g_xy_.data()[idx1] = g.g_xy()[idx2];
                            g_yy_.data()[idx1] = g.g_yy()[idx2];
                            g_pp_.data()[idx1] = g.g_pp()[idx2];
                            vol_.data()[idx1] = g.vol()[idx2];
                            vol2d_.data()[idx1] = g.perpVol()[idx2];
                        }
    }

    //these are for the Field class

    perpendicular_grid perp_grid() const { return perpendicular_grid(*this);}

    const dg::MPI_Vector<thrust::host_vector<double> >& r()const{return r_;}
    const dg::MPI_Vector<thrust::host_vector<double> >& z()const{return z_;}
    const dg::MPI_Vector<thrust::host_vector<double> >& xr()const{return xr_;}
    const dg::MPI_Vector<thrust::host_vector<double> >& yr()const{return yr_;}
    const dg::MPI_Vector<thrust::host_vector<double> >& xz()const{return xz_;}
    const dg::MPI_Vector<thrust::host_vector<double> >& yz()const{return yz_;}
    const dg::MPI_Vector<LocalContainer>& g_xx()const{return g_xx_;}
    const dg::MPI_Vector<LocalContainer>& g_yy()const{return g_yy_;}
    const dg::MPI_Vector<LocalContainer>& g_xy()const{return g_xy_;}
    const dg::MPI_Vector<LocalContainer>& g_pp()const{return g_pp_;}
    const dg::MPI_Vector<LocalContainer>& vol()const{return vol_;}
    const dg::MPI_Vector<LocalContainer>& perpVol()const{return vol2d_;}
    private:
    dg::MPI_Vector<thrust::host_vector<double> > r_, z_, xr_, xz_, yr_, yz_; //3d vector
    dg::MPI_Vector<LocalContainer> g_xx_, g_xy_, g_yy_, g_pp_, vol_, vol2d_;
};

/**
 * @tparam LocalContainer Vector class that holds metric coefficients
 */
template<class LocalContainer>
struct OrthogonalMPIGrid2d : public dg::MPIGrid2d
{
    typedef dg::OrthogonalTag metric_category; 

    template< class Generator>
    OrthogonalMPIGrid2d( const Generator& generator, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx, MPI_Comm comm2d): 
        dg::MPIGrid2d( 0, generator.width(), 0., generator.height(), n, Nx, Ny, bcx, dg::PER, comm2d),
        r_(dg::evaluate( dg::one, *this)), z_(r_), xr_(r_), xz_(r_), yr_(r_), yz_(r_), 
        g_xx_(r_), g_xy_(g_xx_), g_yy_(g_xx_), vol2d_(g_xx_)
    {
        dg::OrthogonalGrid2d<LocalContainer> g( generator, n,Nx, Ny, bcx);
        //divide and conquer
        int dims[2], periods[2], coords[2];
        MPI_Cart_get( communicator(), 2, dims, periods, coords);
        init_X_boundaries( g.x0(), g.x1());
            //for( unsigned py=0; py<dims[1]; py++)
                for( unsigned i=0; i<this->n()*this->Ny(); i++)
                    //for( unsigned px=0; px<dims[0]; px++)
                        for( unsigned j=0; j<this->n()*this->Nx(); j++)
                        {
                            unsigned idx1 = i*this->n()*this->Nx() + j;
                            unsigned idx2 = ((coords[1]*this->n()*this->Ny()+i)*dims[0] + coords[0])*this->n()*this->Nx() + j;
                            r_.data()[idx1] = g.r()[idx2];
                            z_.data()[idx1] = g.z()[idx2];
                            xr_.data()[idx1] = g.xr()[idx2];
                            xz_.data()[idx1] = g.xz()[idx2];
                            yr_.data()[idx1] = g.yr()[idx2];
                            yz_.data()[idx1] = g.yz()[idx2];
                            g_xx_.data()[idx1] = g.g_xx()[idx2];
                            g_xy_.data()[idx1] = g.g_xy()[idx2];
                            g_yy_.data()[idx1] = g.g_yy()[idx2];
                            vol2d_.data()[idx1] = g.perpVol()[idx2];
                        }
    }
    OrthogonalMPIGrid2d( const OrthogonalMPIGrid3d<LocalContainer>& g):
        dg::MPIGrid2d( g.global().x0(), g.global().x1(), g.global().y0(), g.global().y1(), g.global().n(), g.global().Nx(), g.global().Ny(), g.global().bcx(), g.global().bcy(), get_reduced_comm( g.communicator() )),
        r_(dg::evaluate( dg::one, *this)), z_(r_), xr_(r_), xz_(r_), yr_(r_), yz_(r_), 
        g_xx_(r_), g_xy_(g_xx_), g_yy_(g_xx_), vol2d_(g_xx_)
    {
        unsigned s = this->size();
        for( unsigned i=0; i<s; i++)
        {
            r_.data()[i]=g.r().data()[i]; 
            z_.data()[i]=g.z().data()[i]; 
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

    const dg::MPI_Vector<thrust::host_vector<double> >& r()const{return r_;}
    const dg::MPI_Vector<thrust::host_vector<double> >& z()const{return z_;}
    const dg::MPI_Vector<thrust::host_vector<double> >& xr()const{return xr_;}
    const dg::MPI_Vector<thrust::host_vector<double> >& yr()const{return yr_;}
    const dg::MPI_Vector<thrust::host_vector<double> >& xz()const{return xz_;}
    const dg::MPI_Vector<thrust::host_vector<double> >& yz()const{return yz_;}
    const dg::MPI_Vector<LocalContainer>& g_xx()const{return g_xx_;}
    const dg::MPI_Vector<LocalContainer>& g_yy()const{return g_yy_;}
    const dg::MPI_Vector<LocalContainer>& g_xy()const{return g_xy_;}
    const dg::MPI_Vector<LocalContainer>& vol()const{return vol2d_;}
    const dg::MPI_Vector<LocalContainer>& perpVol()const{return vol2d_;}
    private:
    MPI_Comm get_reduced_comm( MPI_Comm src)
    {
        MPI_Comm planeComm;
        int remain_dims[] = {true,true,false}; //true true false
        MPI_Cart_sub( src, remain_dims, &planeComm);
        return planeComm;
    }

    dg::MPI_Vector<thrust::host_vector<double> > r_, z_, xr_, xz_, yr_, yz_; //2d vector
    dg::MPI_Vector<LocalContainer> g_xx_, g_xy_, g_yy_, vol2d_;
};
///@}
}//namespace dg

