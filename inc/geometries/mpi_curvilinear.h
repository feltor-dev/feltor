#pragma once

#include <mpi.h>

#include "dg/backend/mpi_evaluation.h"
#include "dg/backend/mpi_grid.h"
#include "dg/backend/mpi_vector.h"
#include "dg/geometry/mpi_base.h"
#include "curvilinear.h"
#include "generator.h"



namespace dg
{
namespace geo
{

///@cond
struct CurvilinearProductMPIGrid3d; 
///@endcond
//
///@addtogroup grids
///@{
/**
 * @brief A two-dimensional MPI grid based on curvilinear coordinates
 */
struct CurvilinearMPIGrid2d : public dg::aMPIGeometry2d
{
    /// @opydoc hide_grid_parameters2d
    /// @param comm a two-dimensional Cartesian communicator
    /// @note the paramateres given in the constructor are global parameters 
    CurvilinearMPIGrid2d( const aGenerator2d& generator, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx, dg::bc bcy, MPI_Comm comm2d): 
        dg::aMPIGeometry2d( 0, generator.width(), 0., generator.height(), n, Nx, Ny, bcx, bcy, comm2d), handle_(generator)
    {
        //generate global 2d grid and then reduce to local 
        CurvilinearGrid2d g(generator, n, Nx, Ny);
        divide_and_conquer(g);
    }
    ///explicit conversion of 3d product grid to the perpendicular grid
    explicit CurvilinearMPIGrid2d( const CurvilinearProductMPIGrid3d& g);

    ///read access to the generator 
    const aGenerator2d& generator() const{return handle_.get();}
    virtual CurvilinearMPIGrid2d* clone()const{return new CurvilinearMPIGrid2d(*this);}
    private:
    virtual void do_set( unsigned new_n, unsigned new_Nx, unsigned new_Ny)
    {
        dg::aMPITopology2d::do_set(new_n, new_Nx, new_Ny);
        CurvilinearGrid2d g( handle_.get(), new_n, new_Nx, new_Ny);
        divide_and_conquer(g);//distribute to processes
    }
    MPI_Comm get_perp_comm( MPI_Comm src)
    {
        MPI_Comm planeComm;
        int remain_dims[] = {true,true,false}; //true true false
        MPI_Cart_sub( src, remain_dims, &planeComm);
        return planeComm;
    }
    void divide_and_conquer(const CurvilinearGrid2d& g_)
    {
        dg::SparseTensor<thrust::host_vector<double> > jacobian=g_.jacobian(); 
        dg::SparseTensor<thrust::host_vector<double> > metric=g_.metric(); 
        std::vector<thrust::host_vector<double> > map = g_.map();
        for( unsigned i=0; i<3; i++)
            for( unsigned j=0; j<3; j++)
            {
                metric_.idx(i,j) = metric.idx(i,j);
                jac_.idx(i,j) = jacobian.idx(i,j);
            }
        for( unsigned i=0; i<jacobian.values().size(); i++)
            jac_.value(i) = global2local( jacobian.value(i), *this);
        for( unsigned i=0; i<metric.values().size(); i++)
            metric_.value(i) = global2local( metric.value(i), *this);
        map_.resize(map.size());
        for( unsigned i=0; i<map.size(); i++)
            map_[i] = global2local( map[i], *this);
    }

    virtual SparseTensor<host_vector> do_compute_jacobian( ) const {
        return jac_;
    }
    virtual SparseTensor<host_vector> do_compute_metric( ) const {
        return metric_;
    }
    virtual std::vector<host_vector > do_compute_map()const{return map_;}
    dg::SparseTensor<host_vector > jac_, metric_;
    std::vector<host_vector > map_;
    dg::Handle<aGenerator2d> handle_;
};

/**
 * This is s 2x1 curvilinear product space MPI grid
 */
struct CurvilinearProductMPIGrid3d : public dg::aMPIGeometry3d
{
    typedef dg::geo::CurvilinearMPIGrid2d perpendicular_grid; //!< the two-dimensional grid
    /// @opydoc hide_grid_parameters3d
    /// @param comm a three-dimensional Cartesian communicator
    /// @note the paramateres given in the constructor are global parameters 
    CurvilinearProductMPIGrid3d( const aGenerator2d& generator, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm): 
        dg::aMPIGeometry3d( 0, generator.width(), 0., generator.height(), 0., 2.*M_PI, n, Nx, Ny, Nz, bcx, bcy, bcz, comm),
        handle_( generator)
    {
        map_.resize(3);
        CurvilinearMPIGrid2d g(generator,n,Nx,Ny, bcx, bcy, get_perp_comm(comm));
        constructPerp( g);
        constructParallel(this->Nz());
    }

    /*!
     * @brief The grid made up by the first two dimensions in space and process topology
     *
     * This is possible because the 3d grid is a product grid of a 2d perpendicular grid and a 1d parallel grid
     * @return A newly constructed perpendicular grid with the perpendicular communicator
     */
    perpendicular_grid perp_grid() const { return perpendicular_grid(*this);}

    ///read access to the generator
    const aGenerator2d& generator() const{return handle_.get();}
    virtual CurvilinearProductMPIGrid3d* clone()const{return new CurvilinearProductMPIGrid3d(*this);}
    private:
    MPI_Comm get_perp_comm( MPI_Comm src)
    {
        MPI_Comm planeComm;
        int remain_dims[] = {true,true,false}; //true true false
        MPI_Cart_sub( src, remain_dims, &planeComm);
        return planeComm;
    }
    virtual void do_set( unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz)
    {
        dg::aMPITopology3d::do_set(new_n, new_Nx, new_Ny, new_Nz);
        if( !( new_n == n() && new_Nx == Nx() && new_Ny == Ny() ) )
        {
            CurvilinearMPIGrid2d g(handle_.get(),new_n,new_Nx,new_Ny, this->bcx(), this->bcy(), get_perp_comm(communicator()));
            constructPerp( g);
        }
        constructParallel(this->Nz());
    }
    void constructPerp( CurvilinearMPIGrid2d& g2d)
    {
        jac_=g2d.jacobian();
        map_=g2d.map();
    }
    void constructParallel( unsigned localNz )
    {
        map_[2]=dg::evaluate(dg::cooZ3d, *this);
        unsigned size = this->size();
        unsigned size2d = this->n()*this->n()*this->Nx()*this->Ny();
        //resize for 3d values
        for( unsigned r=0; r<4;r++)
        {
            jac_.value(r).data().resize(size);
            jac_.value(r).communicator() = communicator();
        }
        map_[0].data().resize(size); 
        map_[0].communicator() = communicator();
        map_[1].data().resize(size);
        map_[1].communicator() = communicator();
        //lift to 3D grid
        for( unsigned k=1; k<localNz; k++)
            for( unsigned i=0; i<size2d; i++)
            {
                for(unsigned r=0; r<4; r++)
                    jac_.value(r).data()[k*size2d+i] = jac_.value(r).data()[(k-1)*size2d+i];
                map_[0].data()[k*size2d+i] = map_[0].data()[(k-1)*size2d+i];
                map_[1].data()[k*size2d+i] = map_[1].data()[(k-1)*size2d+i];
            }
    }
    virtual SparseTensor<host_vector> do_compute_jacobian( ) const {
        return jac_;
    }
    virtual SparseTensor<host_vector> do_compute_metric( ) const {
        thrust::host_vector<double> tempxx( size()), tempxy(size()), tempyy(size()), temppp(size());
        for( unsigned i=0; i<size(); i++)
        {
            tempxx[i] = (jac_.value(0,0).data()[i]*jac_.value(0,0).data()[i]+jac_.value(0,1).data()[i]*jac_.value(0,1).data()[i]);
            tempxy[i] = (jac_.value(0,0).data()[i]*jac_.value(1,0).data()[i]+jac_.value(0,1).data()[i]*jac_.value(1,1).data()[i]);
            tempyy[i] = (jac_.value(1,0).data()[i]*jac_.value(1,0).data()[i]+jac_.value(1,1).data()[i]*jac_.value(1,1).data()[i]);
            temppp[i] = 1./map_[0].data()[i]/map_[0].data()[i]; //1/R^2
        }
        SparseTensor<host_vector > metric;
        metric.idx(0,0) = 0; metric.value(0) = host_vector(tempxx, communicator());
        metric.idx(1,1) = 1; metric.value(1) = host_vector(tempyy, communicator());
        metric.idx(2,2) = 2; metric.value(2) = host_vector(temppp, communicator());
        if( !handle_.get().isOrthogonal())
        {
            metric.idx(0,1) = metric.idx(1,0) = 3; 
            metric.value(3) = host_vector(tempxy, communicator());
        }
        return metric;
    }
    virtual std::vector<host_vector > do_compute_map()const{return map_;}
    dg::SparseTensor<host_vector > jac_;
    std::vector<host_vector > map_;
    Handle<dg::geo::aGenerator2d> handle_;
};
///@cond
CurvilinearMPIGrid2d::CurvilinearMPIGrid2d( const CurvilinearProductMPIGrid3d& g):
    dg::aMPIGeometry2d( g.global().x0(), g.global().x1(), g.global().y0(), g.global().y1(), g.global().n(), g.global().Nx(), g.global().Ny(), g.global().bcx(), g.global().bcy(), get_perp_comm( g.communicator() )),
    handle_(g.generator())
{
    map_=g.map();
    jac_=g.jacobian();
    metric_=g.metric();
    //now resize to 2d
    unsigned s = this->size();
    for( unsigned i=0; i<jac_.values().size(); i++)
        jac_.value(i).data().resize(s);
    for( unsigned i=0; i<metric_.values().size(); i++)
        metric_.value(i).data().resize(s);
    for( unsigned i=0; i<map_.size(); i++)
        map_[i].data().resize(s);
}
///@endcond

///@}
}//namespace geo
}//namespace dg

