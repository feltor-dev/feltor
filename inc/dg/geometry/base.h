#pragma once

#include "../backend/grid.h"
#include "tensor.h"

namespace dg
{

///@addtogroup geometry
///@{

/**
 * @brief This is the abstract interface class for a two-dimensional Geometry
 */
struct aGeometry2d : public aTopology2d
{
    SharedContainers<thrust::host_vector<double> > jacobian()const{
        return do_provide_jacobian();
    }
    SharedContainers<thrust::host_vector<double> > metric()const { 
        return do_provide_metric(); 
    }
    std::vector<thrust::host_vector<double> > map(){
        return do_provide_map();
    }
    ///Geometries are cloneable
    virtual aGeometry2d* clone()const=0;
    ///allow deletion through base class pointer
    virtual ~aGeometry2d(){}
    protected:
    /*!
     * @copydoc aTopology2d::aTopology2d()
     * @note the default coordinate map will be the identity 
     */
    aGeometry2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy):aTopology2d( x0,x1,y0,y1,n,Nx,Ny,bcx,bcy),map_(3){}
    aGeometry2d( const aGeometry2d& src):map_(src.map_), metric_(src.metric_){}
    aGeometry2d& operator=( const aGeometry2d& src){
        map_=src.map_;
        metric_=src.metric_;
    }
    private:
    virtual SharedContainers<thrust::host_vector<double> > do_compute_metric()const=0;
    virtual SharedContainers<thrust::host_vector<double> > do_compute_jacobian()const=0;
    virtual std::vector<thrust::host_vector<double> > do_compute_map()const=0;
};

/**
 * @brief This is the abstract interface class for a two-dimensional Geometry
 */
struct aGeometry3d : public aTopology3d
{
    std::vector<thrust::host_vector<double> >& map(){return map_;}
    SharedContainers<thrust::host_vector<double> > metric() {
        return do_compute_metric();
    }
    ///Geometries are cloneable
    virtual aGeometry3d* clone()const=0;
    ///allow deletion through base class pointer
    virtual ~aGeometry3d(){}
    protected:
    /*!
     * @copydoc aTopology3d::aTopology3d()
     * @note the default coordinate map will be the identity 
     */
    aGeometry3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz): aTopology3d(x0,x1,y0,y1,z0,z1,n,Nx,Ny,Nz,bcx,bcy,bcz){}
    aGeometry3d( const aGeometry3d& src):map_(src.map_){}
    aGeometry3d& operator=( const aGeometry3d& src){
        map_=src.map_;
    }
    SharedContainers<thrust::host_vector<double> >& map(){return map_;}
    private:
    SharedContainers<thrust::host_vector<double> > map_;
    virtual SharedContainers<thrust::host_vector<double> > do_compute_metric()const=0;
};

///@}
namespace create
{
///@addtogroup metric
///@{
SharedContainers<thrust::host_vector<double> > metric( const aGeometry2d& g)
{
    return g.metric();
}
SharedContainers<thrust::host_vector<double> > metric( const aGeometry3d& g)
{
    return g.metric();
}
///@}

}//namespace create

///@addtogroup basicgrids
///@{

/**
 * @brief two-dimensional Grid with Cartesian metric
 */
struct CartesianGrid2d: public dg::aGeometry2d
{
    ///@copydoc Grid2d::Grid2d()
    CartesianGrid2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx = PER, bc bcy = PER): dg::aGeometry2d(x0,x1,y0,y1,n,Nx,Ny,bcx,bcy){}
    /**
     * @brief Construct from existing topology
     *
     * @param grid existing grid class
     */
    CartesianGrid2d( const dg::Grid2d& grid):dg::Grid2d(grid){}
    virtual CartesianGrid2d* clone()const{return new CartesianGrid2d(*this);}
    private:
    virtual SharedContainers<thrust::host_vector<double> > do_compute_metric()const{
        return SharedContainers<thrust::host_vector<double> >( 3);
    }
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny){
        aTopology2d::do_set(new_n,new_Nx,new_Ny);
    }
};

/**
 * @brief three-dimensional Grid with Cartesian metric
 */
struct CartesianGrid3d: public dg::aGeometry3d
{
    ///@copydoc Grid3d::Grid3d()
    CartesianGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): dg::aGeometry3d(x0,x1,y0,y1,z0,z1,n,Nx,Ny,Nz,bcx,bcy,bcz){}
    /**
     * @brief Implicit type conversion from Grid3d
     * @param grid existing grid class
     */
    CartesianGrid3d( const dg::Grid3d& grid):dg::Grid3d(grid){}
    virtual CartesianGrid3d* clone()const{return new CartesianGrid3d(*this);}
    private:
    virtual SharedContainers<thrust::host_vector<double> > do_compute_metric()const{
        return SharedContainers<thrust::host_vector<double> >( 3);
    }
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz){
        aTopology2d::do_set(new_n,new_Nx,new_Ny,new_Nz);
    }
};

/**
 * @brief three-dimensional Grid with Cylindrical metric
 */
struct CylindricalGrid3d: public dg::aGeometry3d
{
    CylindricalGrid3d( double R0, double R1, double Z0, double Z1, double phi0, double phi1, unsigned n, unsigned NR, unsigned NZ, unsigned Nphi, bc bcR, bc bcZ, bc bcphi = PER): dg::aGeometry3d(R0,R1,Z0,Z1,phi0,phi1,n,NR,NZ,Nphi,bcR,bcZ,bcphi){}
    virtual CylindricalGrid3d* clone()const{return new CylindricalGrid3d(*this);}
    private:
    virtual SharedContainers<thrust::host_vector<double> > do_compute_metric()const{

        std::vector<thrust::host_vector<double> > values(2, size());
        thrust::host_vector<double> R = dg::evaluate(dg::coo1, *this);
        unsigned size2d = n()*n()*Nx()*Ny();
        for( unsigned i = 0; i<Nz(); i++)
        for( unsigned j = 0; j<size2d; j++)
        {
            unsigned idx = i*size2d+j;
            values[1][idx] = R[j];
            values[0][idx] = 1./R[j]/R[j];
        }
        Operator<int> mat_idx(3,-1); mat_idx(2,2) = 0;
        std::vector<int> vec_idx(3,-1); vec_idx[0] = 1;
        return SharedContainers<thrust::host_vector<double> >( mat_idx, vec_idx, values);
    }
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz){
        aTopology2d::do_set(new_n,new_Nx,new_Ny,new_Nz);
    }
};
///@}
} //namespace dg
