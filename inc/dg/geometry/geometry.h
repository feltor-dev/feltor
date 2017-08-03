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
} //namespace dg
