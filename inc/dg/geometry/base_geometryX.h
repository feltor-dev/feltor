#pragma once

#include "../backend/gridX.h"
#include "tensor.h"

namespace dg
{

///@addtogroup basicgeometry
///@{

/**
 * @brief This is the abstract interface class for a two-dimensional GeometryX
 */
struct aGeometryX2d : public aTopologyX2d
{
    ///@copydoc aGeometry2d::jacobian()
    SparseTensor<thrust::host_vector<double> > jacobian()const{
        return do_compute_jacobian();
    }
    ///@copydoc aGeometry2d::metric()
    SparseTensor<thrust::host_vector<double> > metric()const { 
        return do_compute_metric(); 
    }
    ///@copydoc aGeometry2d::map()
    std::vector<thrust::host_vector<double> > map()const{
        return do_compute_map();
    }
    ///Geometries are cloneable
    virtual aGeometryX2d* clone()const=0;
    ///allow deletion through base class pointer
    virtual ~aGeometryX2d(){}
    protected:
    /*!
     * @copydoc aTopologyX2d::aTopologyX2d()
     * @note the default coordinate map will be the identity 
     */
    aGeometryX2d( double x0, double x1, double y0, double y1, double fx, double fy, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy):aTopologyX2d( x0,x1,y0,y1,fx,fy,n,Nx,Ny,bcx,bcy){}
    ///@copydoc aTopologyX2d::aTopologyX2d(const aTopologyX2d&)
    aGeometryX2d( const aGeometryX2d& src):aTopologyX2d(src){}
    ///@copydoc aTopologyX2d::operator=(const aTopologyX2d&)
    aGeometryX2d& operator=( const aGeometryX2d& src){
        aTopologyX2d::operator=(src);
        return *this;
    }
    private:
    virtual SparseTensor<thrust::host_vector<double> > do_compute_metric()const {
        return SparseTensor<thrust::host_vector<double> >();
    }
    virtual SparseTensor<thrust::host_vector<double> > do_compute_jacobian()const {
        return SparseTensor<thrust::host_vector<double> >();
    }
    virtual std::vector<thrust::host_vector<double> > do_compute_map()const{
        std::vector<thrust::host_vector<double> > map(2);
        map[0] = dg::evaluate(dg::cooX2d, *this);
        map[1] = dg::evaluate(dg::cooY2d, *this);
        return map;
    }


};

/**
 * @brief This is the abstract interface class for a three-dimensional GeometryX
 */
struct aGeometryX3d : public aTopologyX3d
{
    ///@copydoc aGeometry3d::jacobian()
    SparseTensor<thrust::host_vector<double> > jacobian()const{
        return do_compute_jacobian();
    }
    ///@copydoc aGeometry3d::metric()
    SparseTensor<thrust::host_vector<double> > metric()const { 
        return do_compute_metric(); 
    }
    ///@copydoc aGeometry3d::map()
    std::vector<thrust::host_vector<double> > map()const{
        return do_compute_map();
    }
    ///Geometries are cloneable
    virtual aGeometryX3d* clone()const=0;
    ///allow deletion through base class pointer
    virtual ~aGeometryX3d(){}
    protected:
    /*!
     * @copydoc aTopologyX3d::aTopologyX3d()
     * @note the default coordinate map will be the identity 
     */
    aGeometryX3d( double x0, double x1, double y0, double y1, double z0, double z1, double fx, double fy, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz): aTopologyX3d(x0,x1,y0,y1,z0,z1,fx,fy,n,Nx,Ny,Nz,bcx,bcy,bcz){}
    ///@copydoc aTopologyX3d::aTopologyX3d(const aTopologyX3d&)
    aGeometryX3d( const aGeometryX3d& src):aTopologyX3d(src){}
    ///@copydoc aTopologyX3d::operator=(const aTopologyX3d&)
    aGeometryX3d& operator=( const aGeometryX3d& src){
        aTopologyX3d::operator=(src);
        return *this;
    }
    private:
    virtual SparseTensor<thrust::host_vector<double> > do_compute_metric()const {
        return SparseTensor<thrust::host_vector<double> >();
    }
    virtual SparseTensor<thrust::host_vector<double> > do_compute_jacobian()const {
        return SparseTensor<thrust::host_vector<double> >();
    }
    virtual std::vector<thrust::host_vector<double> > do_compute_map()const{
        std::vector<thrust::host_vector<double> > map(3);
        map[0] = dg::evaluate(dg::cooX3d, *this);
        map[1] = dg::evaluate(dg::cooY3d, *this);
        map[2] = dg::evaluate(dg::cooZ3d, *this);
        return map;
    }
};

///@}

///@addtogroup geometry
///@{

/**
 * @brief two-dimensional GridX with Cartesian metric
 */
struct CartesianGridX2d: public dg::aGeometryX2d
{
    ///@copydoc GridX2d::GridX2d()
    CartesianGridX2d( double x0, double x1, double y0, double y1, double fx, double fy, unsigned n, unsigned Nx, unsigned Ny, bc bcx = PER, bc bcy = PER): dg::aGeometryX2d(x0,x1,y0,y1,fx,fy,n,Nx,Ny,bcx,bcy){}
    /**
     * @brief Construct from existing topology
     * @param g existing grid class
     */
    CartesianGridX2d( const dg::GridX2d& g):dg::aGeometryX2d(g.x0(),g.x1(),g.y0(),g.y1(),g.fx(),g.fy(),g.n(),g.Nx(),g.Ny(),g.bcx(),g.bcy()){}
    virtual CartesianGridX2d* clone()const{return new CartesianGridX2d(*this);}
};

/**
 * @brief three-dimensional GridX with Cartesian metric
 */
struct CartesianGridX3d: public dg::aGeometryX3d
{
    ///@copydoc GridX3d::GridX3d()
    CartesianGridX3d( double x0, double x1, double y0, double y1, double z0, double z1, double fx, double fy, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): dg::aGeometryX3d(x0,x1,y0,y1,z0,z1,fx,fy,n,Nx,Ny,Nz,bcx,bcy,bcz){}
    /**
     * @brief Implicit type conversion from GridX3d
     * @param g existing grid class
     */
    CartesianGridX3d( const dg::GridX3d& g):dg::aGeometryX3d(g.x0(), g.x1(), g.y0(), g.y1(), g.z0(), g.z1(),g.fx(),g.fy(),g.n(),g.Nx(),g.Ny(),g.Nz(),g.bcx(),g.bcy(),g.bcz()){}
    virtual CartesianGridX3d* clone()const{return new CartesianGridX3d(*this);}
};

///@}

///@copydoc pullback(const Functor&,const aGeometry2d&)
///@ingroup pullback
template< class Functor>
thrust::host_vector<double> pullback( const Functor& f, const aGeometryX2d& g)
{
    std::vector<thrust::host_vector<double> > map = g.map();
    thrust::host_vector<double> vec( g.size());
    for( unsigned i=0; i<g.size(); i++)
        vec[i] = f( map[0][i], map[1][i]);
    return vec;
}

///@copydoc pullback(const Functor&,const aGeometry2d&)
///@ingroup pullback
template< class Functor>
thrust::host_vector<double> pullback( const Functor& f, const aGeometryX3d& g)
{
    std::vector<thrust::host_vector<double> > map = g.map();
    thrust::host_vector<double> vec( g.size());
    for( unsigned i=0; i<g.size(); i++)
        vec[i] = f( map[0][i], map[1][i], map[2][i]);
    return vec;
}
} //namespace dg
