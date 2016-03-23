#pragma once

#include <mpi.h>

#include "orthogonal.h"
#include "dg/backend/mpi_grid.h"
#include "dg/backend/mpi_vector.h"



namespace orthogonal
{

///@cond
template< class container>
struct MPIRingGrid2d; 
///@endcond

/**
 * @brief A three-dimensional grid based on "almost-orthogonal" coordinates by Ribeiro and Scott 2010 (MPI Version)
 *
 * @tparam container Vector class that holds metric coefficients
 */
template<class LocalContainer>
struct MPIRingGrid3d : public dg::MPI_Grid3d
{
    typedef dg::CurvilinearCylindricalTag metric_category; //!< metric tag
    typedef MPIRingGrid2d<LocalContainer> perpendicular_grid; //!< the two-dimensional grid

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
     * @param comm The mpi communicator class
     */
    MPIRingGrid3d( solovev::GeomParameters gp, double psi_0, double psi_1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx, MPI_Comm comm): 
        dg::MPI_Grid3d( 0, 1, 0., 2*M_PI, 0., 2.*M_PI, n, Nx, Ny, Nz, bcx, dg::PER, dg::PER, comm),
        f_( dg::evaluate( dg::one, *this)), g_(f_), r_(f_), z_(r_), xr_(r_), xz_(r_), yr_(r_), yz_(r_),
        g_xx_(r_), g_xy_(g_xx_), g_yy_(g_xx_), g_pp_(g_xx_), vol_(g_xx_), vol2d_(g_xx_)
    {
        RingGrid3d<LocalContainer> g( gp, psi_0, psi_1, n,Nx, Ny, local().Nz(), bcx);
        f_x_ = g.f1_x();
        f2_xy_ = g.f2_xy();

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
                            f_.data()[idx1] = g.f1()[idx2];
                            g_.data()[idx1] = g.f2()[idx2];
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

    /**
     * @brief The f_1 vector over the global grid
     *
     * @return 
     */
    const thrust::host_vector<double>& f1_x()const{return f_x_;}
    /**
     * @brief The f_2 vector over the global grid
     *
     * @return 
     */
    const thrust::host_vector<double>& f2_xy()const{return f2_xy_;}
    /**
     * @brief The f_1 vector over the local grid
     *
     * @return 
     */
    const dg::MPI_Vector<thrust::host_vector<double> >& f1()const{return f_;}
    /**
     * @brief The f_2 vector over the local grid
     *
     * @return 
     */
    const dg::MPI_Vector<thrust::host_vector<double> >& f2()const{return g_;}
    perpendicular_grid perp_grid() const { return MPIRingGrid2d<LocalContainer>(*this);}

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
    thrust::host_vector<double> f_x_; //1d vector
    thrust::host_vector<double> f2_xy_; //2d vector
    dg::MPI_Vector<thrust::host_vector<double> > f_, g_, r_, z_, xr_, xz_, yr_, yz_; //3d vector
    dg::MPI_Vector<LocalContainer> g_xx_, g_xy_, g_yy_, g_pp_, vol_, vol2d_;
};

/**
 * @brief A two-dimensional grid based on "almost-orthogonal" coordinates by Ribeiro and Scott 2010
 */
template<class LocalContainer>
struct MPIRingGrid2d : public dg::MPI_Grid2d
{
    typedef dg::CurvilinearCylindricalTag metric_category; 

    /**
     * @brief Construct 
     *
     * @param gp The geometric parameters define the magnetic field
     * @param psi_0 lower boundary for psi
     * @param psi_1 upper boundary for psi
     * @param n The dG number of polynomials
     * @param Nx The number of points in x-direction
     * @param Ny The number of points in y-direction
     * @param bcx The boundary condition in x (y,z are periodic)
     * @param comm2d The 2d mpi communicator class
     */
    MPIRingGrid2d( solovev::GeomParameters gp, double psi_0, double psi_1, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx, MPI_Comm comm2d): 
        dg::MPI_Grid2d( 0, 1, 0., 2*M_PI, n, Nx, Ny, bcx, dg::PER, comm2d),
        f_( dg::evaluate( dg::one, *this)), g_(f_), r_(f_), z_(r_), xr_(r_), xz_(r_), yr_(r_), yz_(r_),
        g_xx_(r_), g_xy_(g_xx_), g_yy_(g_xx_), vol2d_(g_xx_)
    {
        RingGrid2d<LocalContainer> g( gp, psi_0, psi_1, n,Nx, Ny, bcx);
        f_x_ = g.f1_x();
        f2_xy_ = g.f2_xy();
        //divide and conquer
        int dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        init_X_boundaries( g.x0(), g.x1());
            //for( unsigned py=0; py<dims[1]; py++)
                for( unsigned i=0; i<this->n()*this->Ny(); i++)
                    //for( unsigned px=0; px<dims[0]; px++)
                        for( unsigned j=0; j<this->n()*this->Nx(); j++)
                        {
                            unsigned idx1 = i*this->n()*this->Nx() + j;
                            unsigned idx2 = ((coords[1]*this->n()*this->Ny()+i)*dims[0] + coords[0])*this->n()*this->Nx() + j;
                            f_.data()[idx1] = g.f1()[idx2];
                            g_.data()[idx1] = g.f2()[idx2];
                            r_.data()[idx1] = g.r()[idx2];
                            z_.data()[idx1] = g.z()[idx2];
                            xr_.data()[idx1] = g.xr()[idx2];
                            xz_.data()[idx1] = g.xz()[idx2];
                            yr_.data()[idx1] = g.yr()[idx2];
                            yz_.data()[idx1] = g.yz()[idx2];
                            g_xx_.data()[idx1] = g.g_xx()[idx2];
                            g_xy_.data()[idx1] = g.g_xy()[idx2];
                            g_yy_.data()[idx1] = g.g_yy()[idx2];
                            vol2d_.data()[idx1] = g.vol2d()[idx2];
                        }
    }
    MPIRingGrid2d( const MPIRingGrid3d<LocalContainer>& g):
        dg::MPI_Grid2d( g.global().x0(), g.global().x1(), g.global().y0(), g.global().y1(), g.global().n(), g.global().Nx(), g.global().Ny(), g.global().bcx(), g.global().bcy(), get_reduced_comm( g.communicator() )),
        f_( dg::evaluate( dg::one, *this)), g_(f_), r_(f_), z_(r_), xr_(r_), xz_(r_), yr_(r_), yz_(r_),
        g_xx_(r_), g_xy_(g_xx_), g_yy_(g_xx_), vol2d_(g_xx_)
    {
        f_x_ = g.f1_x();
        f2_xy_ = g.f2_xy();
        unsigned s = this->size();
        for( unsigned i=0; i<s; i++)
        {
            r_.data()[i]=g.r().data()[i]; 
            f_.data()[i]=g.f1().data()[i]; 
            g_.data()[i]=g.f2().data()[i]; 
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

    /**
     * @brief 1D version of the f_1 vector over the global 2d grid
     *
     * @return 
     */
    const thrust::host_vector<double>& f1_x()const{return f_x_;}
    /**
     * @brief 2D version of the f_2 vector over the global 2D grid
     *
     * @return 
     */
    const thrust::host_vector<double>& f2_xy()const{return f2_xy_;}
    /**
     * @brief Get the whole f_1 vector
     *
     * @return 
     */
    const dg::MPI_Vector<thrust::host_vector<double> >& f1()const{return f_;}
    /**
     * @brief Get the whole f_2 vector
     *
     * @return 
     */
    const dg::MPI_Vector<thrust::host_vector<double> >& f2()const{return g_;}

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
    thrust::host_vector<double> f_x_, f2_xy_; 
    dg::MPI_Vector<thrust::host_vector<double> > f_, g_; 

    dg::MPI_Vector<thrust::host_vector<double> > r_, z_, xr_, xz_, yr_, yz_; //2d vector
    dg::MPI_Vector<LocalContainer> g_xx_, g_xy_, g_yy_, vol2d_;
};

}//namespace orthogonal

namespace dg{
/**
 * @brief This function pulls back a function defined in cartesian coordinates R,Z to the orthogonal coordinates x,y,\phi
 *
 * i.e. F(x,y) = f(R(x,y), Z(x,y))
 * @tparam BinaryOp The function object 
 * @param f The function defined on R,Z
 * @param g The grid
 *
 * @return A set of points representing F(x,y)
 */
template< class BinaryOp, class LocalContainer>
MPI_Vector<thrust::host_vector<double> > pullback( BinaryOp f, const orthogonal::MPIRingGrid2d<LocalContainer>& g)
{
    thrust::host_vector<double> vec( g.size());
    for( unsigned i=0; i<g.size(); i++)
        vec[i] = f( g.r().data()[i], g.z().data()[i]);
    MPI_Vector<thrust::host_vector<double> > v( vec, g.communicator());
    return v;
}
///@cond
template<class LocalContainer>
MPI_Vector<thrust::host_vector<double> > pullback( double(f)(double,double), const orthogonal::MPIRingGrid2d<LocalContainer>& g)
{
    return pullback<double(double,double),LocalContainer>( f, g);
}
///@endcond
/**
 * @brief This function pulls back a function defined in cylindrical coordinates R,Z,\phi to the orthogonal coordinates x,y,\phi
 *
 * i.e. F(x,y,\phi) = f(R(x,y), Z(x,y), \phi)
 * @tparam TernaryOp The function object 
 * @param f The function defined on R,Z,\phi
 * @param g The grid
 *
 * @return A set of points representing F(x,y,\phi)
 */
template< class TernaryOp, class LocalContainer>
MPI_Vector<thrust::host_vector<double> > pullback( TernaryOp f, const orthogonal::MPIRingGrid3d<LocalContainer>& g)
{
    thrust::host_vector<double> vec( g.size());
    unsigned size2d = g.n()*g.n()*g.Nx()*g.Ny();
    Grid1d<double> gz( g.z0(), g.z1(), 1, g.Nz());
    thrust::host_vector<double> absz = create::abscissas( gz);
    for( unsigned k=0; k<g.Nz(); k++)
        for( unsigned i=0; i<size2d; i++)
            vec[k*size2d+i] = f( g.r().data()[k*size2d+i], g.z().data()[k*size2d+i], absz[k]);
    MPI_Vector<thrust::host_vector<double> > v( vec, g.communicator());
    return v;
}
///@cond
template<class LocalContainer>
MPI_Vector<thrust::host_vector<double> > pullback( double(f)(double,double,double), const orthogonal::MPIRingGrid3d<LocalContainer>& g)
{
    return pullback<double(double,double,double),LocalContainer>( f, g);
}
///@endcond
//

}//namespace dg
