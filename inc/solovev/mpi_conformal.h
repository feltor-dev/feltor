#pragma once

#include "conformal.h"
#include "dg/backend/mpi_grid.h"



namespace solovev
{



template<class container>
class ConformalMPIRingGrid : public MPI_Grid3d
{
    typedef CurvilinearCylindricalTag metric_category; 

    ConformalMPIRingGrid( GeomParameters gp, double psi_0, double psi_1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx, MPI_Comm comm): 
        MPI_Grid3d( 0, detail::Fpsi(gp, psi_0).find_x1( psi_1), 0., 2*M_PI, 0., 2.*M_PI, n, Nx, Ny, Nz, bcx, dg::PER, dg::PER, comm),
        r_( dg::evaluate( dg::one, *this)), z_(r_), xr_(r_), xz_(r_), yr_(r_), yz_(r_),
        g_xx_(r_), g_xy_(g_xx_), g_yy_(g_xx_), g_pp_(g_xx_), vol_(g_xx_), vol2d_(g_xx_)
    {
        ConformalRingGrid g( gp, psi_0, psi_1, n,Nx, Ny, local().Nz(), bcx);
        //divide and conquer
        f_x_ = g.f_x();
        r_.data() = g.r();
        z_.data() = g.z();
        xr_.data() = g.xr();
        yr_.data() = g.xz();
        xz_.data() = g.yr();
        yz_.data() = g.yz();

        g_xx_.data() = g.g_xx();
        g_xy_.data() = g.g_xy();
        g_yy_.data() = g.g_yy();
        g_pp_.data() = g.g_pp();
        vol_.data() = g.vol();
        vol2d_.data() = g.vol2d();

    }


    private:
    thrust::host_vector<double> f_x_; //1d vector
    MPI_Vector<thrust::host_vector<double> > r_, z_, xr_, xz_, yr_, yz_; //3d vector
    MPI_Vector<container> g_xx_, g_xy_, g_yy_, g_pp_, vol_, vol2d_;

}

}//namespace solovev
