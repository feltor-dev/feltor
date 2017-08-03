#pragma once

#include "dg/backend/grid.h"
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
    const SharedContainers<thrust::host_vector<double> >& map()const{return map_;}
    SharedContainers<thrust::host_vector<double> > compute_metric()const {
        return do_compute_metric();
    }
    ///Geometries are cloneable
    virtual aGeometry2d* clone()const=0;
    ///allow deletion through base class pointer
    virtual ~aGeometry2d(){}
    protected:
    aGeometry2d(const SharedContainers<thrust::host_vector<double> >& map, const SharedContainers<container >& metric): map_(map), metric_(metric){}
    aGeometry2d( const aGeometry2d& src):map_(src.map_), metric_(src.metric_){}
    aGeometry2d& operator=( const aGeometry2d& src){
        map_=src.map_;
        metric_=src.metric_;
    }
    SharedContainers<thrust::host_vector<double> >& map(){return map_;}
    private:
    SharedContainers<thrust::host_vector<double> > map_;
    virtual SharedContainers<thrust::host_vector<double> > do_compute_metric()const=0;
};

/**
 * @brief This is the abstract interface class for a two-dimensional Geometry
 */
struct aGeometry3d : public aTopology3d
{
    const SharedContainers<thrust::host_vector<double> >& map()const{return map_;}
    SharedContainers<thrust::host_vector<double> > compute_metric()const {
        return do_compute_metric();
    }
    ///Geometries are cloneable
    virtual aGeometry3d* clone()const=0;
    ///allow deletion through base class pointer
    virtual ~aGeometry3d(){}
    protected:
    aGeometry3d(const SharedContainers<thrust::host_vector<double> >& map): map_(map){}
    aGeometry3d( const aGeometry2d& src):map_(src.map_){}
    aGeometry3d& operator=( const aGeometry3d& src){
        map_=src.map_;
    }
    SharedContainers<thrust::host_vector<double> >& map(){return map_;}
    private:
    SharedContainers<thrust::host_vector<double> > map_;
    virtual SharedContainers<thrust::host_vector<double> > do_compute_metric()const=0;
};

namespace create
{

SharedContainers<thrust::host_vector<double> > metric( const aGeometry2d& g)
{
    return g.compute_metric();
}
SharedContainers<thrust::host_vector<double> > metric( const aGeometry3d& g)
{
    return g.compute_metric();
}

}//namespace create

///@}
///@addtogroup basicgrids
///@{

/**
 * @brief two-dimensional Grid with Cartesian metric
 */
struct CartesianGrid2d: public dg::aGeometry2d
{
    typedef OrthonormalTag metric_category; 
    ///@copydoc Grid2d::Grid2d()
    CartesianGrid2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx = PER, bc bcy = PER): dg::Grid2d(x0,x1,y0,y1,n,Nx,Ny,bcx,bcy){}
    /**
     * @brief Construct from existing topology
     *
     * @param grid existing grid class
     */
    CartesianGrid2d( const dg::Grid2d& grid):dg::Grid2d(grid){}
};

/**
 * @brief three-dimensional Grid with Cartesian metric
 */
struct CartesianGrid3d: public dg::aGeometry3d
{
    typedef OrthonormalTag metric_category; 
    ///@copydoc Grid3d::Grid3d()
    CartesianGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): dg::Grid3d(x0,x1,y0,y1,z0,z1,n,Nx,Ny,Nz,bcx,bcy,bcz){}
    /**
     * @brief Construct from existing topology
     *
     * @param grid existing grid class
     */
    CartesianGrid3d( const dg::Grid3d& grid):dg::Grid3d(grid){}
};

///@}
} //namespace dg
