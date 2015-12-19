#pragma once

#include "conformal.h"
#include "dg/backend/mpi_grid.h"



namespace solovev
{

template< class container>
struct ConformalMPIRingGrid2d; 

template<class container>
class ConformalMPIRingGrid3d : public MPI_Grid3d
{
    typedef CurvilinearCylindricalTag metric_category; 
    typedef ConformalMPIRingGrid2d<container> perpendicular_grid;

    ConformalMPIRingGrid( GeomParameters gp, double psi_0, double psi_1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx, MPI_Comm comm): 
        MPI_Grid3d( 0, detail::Fpsi(gp, psi_0).find_x1( psi_1), 0., 2*M_PI, 0., 2.*M_PI, n, Nx, Ny, Nz, bcx, dg::PER, dg::PER, comm),
        r_( dg::evaluate( dg::one, *this)), z_(r_), xr_(r_), xz_(r_), yr_(r_), yz_(r_),
        g_xx_(r_), g_xy_(g_xx_), g_yy_(g_xx_), g_pp_(g_xx_), vol_(g_xx_), vol2d_(g_xx_)
    {
        ConformalRingGrid g( gp, psi_0, psi_1, n,Nx, Ny, local().Nz(), bcx);
        f_x_ = g.f_x();
        //divide and conquer
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        return g.x0() + g.lx()/(double)dims[0]*(double)coords[0]; 
        for( unsigned s=0; s<Nz(); s++)
            //for( unsigned py=0; py<dims[1]; py++)
                for( unsigned i=0; i<n()*Ny(); i++)
                    //for( unsigned px=0; px<dims[0]; px++)
                        for( unsigned j=0; j<n()*Nx(); j++)
                        {
                            unsigned idx1 = (s*n()*Ny()+i)*n()*Nx() + j
                            unsigned idx2 = (((s*dims[1]+coords[1])*n()*Ny()+i)*dims[0] + coords[0])*n()*Nx() + j;
                            r_.data()[idx1] = g.r()[idx2];
                            z_.data()[idx1] = g.z()[idx2];
                            xr_.data()[idx1] = g.xr()[idx2];
                            yr_.data()[idx1] = g.xz()[idx2];
                            xz_.data()[idx1] = g.yr()[idx2];
                            yz_.data()[idx1] = g.yz()[idx2];
                            g_xx_.data()[idx1] = g.g_xx()[idx2];
                            g_xy_.data()[idx1] = g.g_xy()[idx2];
                            g_yy_.data()[idx1] = g.g_yy()[idx2];
                            g_pp_.data()[idx1] = g.g_pp()[idx2];
                            vol_.data()[idx1] = g.vol()[idx2];
                            vol2d_.data()[idx1] = g.vol2d()[idx2];
                        }
    }

    const MPI_Vector<thrust::host_vector<double> >& r()const{return r_;}
    const MPI_Vector<thrust::host_vector<double> >& z()const{return z_;}
    const MPI_Vector<thrust::host_vector<double> >& xr()const{return xr_;}
    const MPI_Vector<thrust::host_vector<double> >& yr()const{return yr_;}
    const MPI_Vector<thrust::host_vector<double> >& xz()const{return xz_;}
    const MPI_Vector<thrust::host_vector<double> >& yz()const{return yz_;}
    const thrust::host_vector<double>& f_x()const{return f_x_;}
    const MPI_Vector<container>& g_xx()const{return g_xx_;}
    const MPI_Vector<container>& g_yy()const{return g_yy_;}
    const MPI_Vector<container>& g_xy()const{return g_xy_;}
    const MPI_Vector<container>& g_pp()const{return g_pp_;}
    const MPI_Vector<container>& vol()const{return vol_;}
    const MPI_Vector<container>& perpVol()const{return vol2d_;}
    perpendicular_grid perp_grid() const { return ConformalRingGrid2d<container>(*this);}
    private:
    thrust::host_vector<double> f_x_; //1d vector
    MPI_Vector<thrust::host_vector<double> > r_, z_, xr_, xz_, yr_, yz_; //3d vector
    MPI_Vector<container> g_xx_, g_xy_, g_yy_, g_pp_, vol_, vol2d_;
};
template<class container>
class ConformalMPIRingGrid2d : public MPI_Grid2d
{
    typedef CurvilinearCylindricalTag metric_category; 

    ConformalMPIRingGrid2d( GeomParameters gp, double psi_0, double psi_1, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx, MPI_Comm comm2d): 
        MPI_Grid2d( 0, detail::Fpsi(gp, psi_0).find_x1( psi_1), 0., 2*M_PI, n, Nx, Ny, bcx, dg::PER, comm2d),
        r_( dg::evaluate( dg::one, *this)), z_(r_), xr_(r_), xz_(r_), yr_(r_), yz_(r_),
        g_xx_(r_), g_xy_(g_xx_), g_yy_(g_xx_), vol2d_(g_xx_)
    {
        ConformalRingGrid2d g( gp, psi_0, psi_1, n,Nx, Ny, bcx);
        f_x_ = g.f_x();
        //divide and conquer
        int dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        return g.x0() + g.lx()/(double)dims[0]*(double)coords[0]; 
            //for( unsigned py=0; py<dims[1]; py++)
                for( unsigned i=0; i<n()*Ny(); i++)
                    //for( unsigned px=0; px<dims[0]; px++)
                        for( unsigned j=0; j<n()*Nx(); j++)
                        {
                            unsigned idx1 = i*n()*Nx() + j
                            unsigned idx2 = ((coords[1]*n()*Ny()+i)*dims[0] + coords[0])*n()*Nx() + j;
                            r_.data()[idx1] = g.r()[idx2];
                            z_.data()[idx1] = g.z()[idx2];
                            xr_.data()[idx1] = g.xr()[idx2];
                            yr_.data()[idx1] = g.xz()[idx2];
                            xz_.data()[idx1] = g.yr()[idx2];
                            yz_.data()[idx1] = g.yz()[idx2];
                            g_xx_.data()[idx1] = g.g_xx()[idx2];
                            g_xy_.data()[idx1] = g.g_xy()[idx2];
                            g_yy_.data()[idx1] = g.g_yy()[idx2];
                            vol2d_.data()[idx1] = g.vol2d()[idx2];
                        }
    }
    ConformalMPIRingGrid2d( const ConformalMPIRingGrid3d& g):
        dg::MPI_Grid2d( g.global().x0(), g.global().x1(), g.global().y0(), g.global().y1(), g.global().n(), g.global().Nx(), g.global().Ny(), g.global().bcx(), g.global().bcy(), get_reduced_comm( g.communicator() )
    {
        f_x_ = g.f_x();
        unsigned s = this->size();
        for( unsigned i=0; i<s; i++)
        {r_.data()[i]=g.r().data()[i], z_.data()[i]=g.z(.data())[i], xr_.data()[i]=g.xr().data()[i], xz_.data()[i]=g.xz().data()[i], yr_.data()[i]=g.yr().data()[i], yz_.data()[i]=g.yz().data()[i];}
        thrust::copy( g.g_xx().data().begin, g.g_xx().data().begin()+s, g_xx.data().begin())
        thrust::copy( g.g_xy().data().begin, g.g_xy().data().begin()+s, g_xy.data().begin())
        thrust::copy( g.g_yy().data().begin, g.g_yy().data().begin()+s, g_yy.data().begin())
        thrust::copy( g.perpVol().data().begin, g.perpVol().data().begin()+s, vol2d_.data().begin())
        
    }

    const MPI_Vector<thrust::host_vector<double> >& r()const{return r_;}
    const MPI_Vector<thrust::host_vector<double> >& z()const{return z_;}
    const MPI_Vector<thrust::host_vector<double> >& xr()const{return xr_;}
    const MPI_Vector<thrust::host_vector<double> >& yr()const{return yr_;}
    const MPI_Vector<thrust::host_vector<double> >& xz()const{return xz_;}
    const MPI_Vector<thrust::host_vector<double> >& yz()const{return yz_;}
    const thrust::host_vector<double>& f_x()const{return f_x_;}
    const MPI_Vector<container>& g_xx()const{return g_xx_;}
    const MPI_Vector<container>& g_yy()const{return g_yy_;}
    const MPI_Vector<container>& g_xy()const{return g_xy_;}
    const MPI_Vector<container>& vol()const{return vol2d_;}
    const MPI_Vector<container>& perpVol()const{return vol2d_;}
    private:
    MPI_Comm get_reduced_comm( MPI_Comm src)
    {
        MPI_Comm planeComm;
        int remain_dims[] = {true,true,false}; //true true false
        MPI_Cart_sub( src, remain_dims, &planeComm);
    }
    thrust::host_vector<double> f_x_; //1d vector
    MPI_Vector<thrust::host_vector<double> > r_, z_, xr_, xz_, yr_, yz_; //2d vector
    MPI_Vector<container> g_xx_, g_xy_, g_yy_, vol2d_;
};

}//namespace solovev
