#pragma once

#include "gridX.h"
#include "evaluationX.cuh"
#include "tensor.h"

namespace dg
{

///@addtogroup basicgeometry
///@{

/**
 * @brief This is the abstract interface class for a two-dimensional BasicGeometryX
 */
template<class real_type>
struct aBasicGeometryX2d : public aBasicTopologyX2d<real_type>
{
    ///@copydoc aBasicGeometry2d::jacobian()
    SparseTensor<thrust::host_vector<real_type> > jacobian()const{
        return do_compute_jacobian();
    }
    ///@copydoc aBasicGeometry2d::metric()
    SparseTensor<thrust::host_vector<real_type> > metric()const {
        return do_compute_metric();
    }
    ///@copydoc aBasicGeometry2d::map()
    std::vector<thrust::host_vector<real_type> > map()const{
        return do_compute_map();
    }
    ///Geometries are cloneable
    virtual aBasicGeometryX2d* clone()const=0;
    ///allow deletion through base class pointer
    virtual ~aBasicGeometryX2d(){}
    protected:
    /*!
     * @copydoc aBasicTopologyX2d::aBasicTopologyX2d()
     * @note the default coordinate map will be the identity
     */
    aBasicGeometryX2d( real_type x0, real_type x1, real_type y0, real_type y1, real_type fx, real_type fy, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy):aBasicTopologyX2d<real_type>( x0,x1,y0,y1,fx,fy,n,Nx,Ny,bcx,bcy){}
    ///@copydoc aBasicTopologyX2d::aBasicTopologyX2d(const aBasicTopologyX2d&)
    aBasicGeometryX2d( const aBasicGeometryX2d& src):aBasicTopologyX2d<real_type>(src){}
    ///@copydoc aBasicTopologyX2d::operator=(const aBasicTopologyX2d&)
    aBasicGeometryX2d& operator=( const aBasicGeometryX2d& src){
        aBasicTopologyX2d<real_type>::operator=(src);
        return *this;
    }
    private:
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_metric()const {
        return SparseTensor<thrust::host_vector<real_type> >();
    }
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_jacobian()const {
        return SparseTensor<thrust::host_vector<real_type> >();
    }
    virtual std::vector<thrust::host_vector<real_type> > do_compute_map()const{
        std::vector<thrust::host_vector<real_type> > map(2);
        map[0] = dg::evaluate(dg::cooX2d, *this);
        map[1] = dg::evaluate(dg::cooY2d, *this);
        return map;
    }


};

/**
 * @brief This is the abstract interface class for a three-dimensional BasicGeometryX
 */
template<class real_type>
struct aBasicGeometryX3d : public aBasicTopologyX3d<real_type>
{
    ///@copydoc aBasicGeometry3d::jacobian()
    SparseTensor<thrust::host_vector<real_type> > jacobian()const{
        return do_compute_jacobian();
    }
    ///@copydoc aBasicGeometry3d::metric()
    SparseTensor<thrust::host_vector<real_type> > metric()const {
        return do_compute_metric();
    }
    ///@copydoc aBasicGeometry3d::map()
    std::vector<thrust::host_vector<real_type> > map()const{
        return do_compute_map();
    }
    ///Geometries are cloneable
    virtual aBasicGeometryX3d* clone()const=0;
    ///allow deletion through base class pointer
    virtual ~aBasicGeometryX3d(){}
    protected:
    /*!
     * @copydoc aBasicTopologyX3d::aBasicTopologyX3d()
     * @note the default coordinate map will be the identity
     */
    aBasicGeometryX3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, real_type fx, real_type fy, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz): aBasicTopologyX3d<real_type>(x0,x1,y0,y1,z0,z1,fx,fy,n,Nx,Ny,Nz,bcx,bcy,bcz){}
    ///@copydoc aBasicTopologyX3d::aBasicTopologyX3d(const aBasicTopologyX3d&)
    aBasicGeometryX3d( const aBasicGeometryX3d& src):aBasicTopologyX3d<real_type>(src){}
    ///@copydoc aBasicTopologyX3d::operator=(const aBasicTopologyX3d&)
    aBasicGeometryX3d& operator=( const aBasicGeometryX3d& src){
        aBasicTopologyX3d<real_type>::operator=(src);
        return *this;
    }
    private:
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_metric()const {
        return SparseTensor<thrust::host_vector<real_type> >();
    }
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_jacobian()const {
        return SparseTensor<thrust::host_vector<real_type> >();
    }
    virtual std::vector<thrust::host_vector<real_type> > do_compute_map()const{
        std::vector<thrust::host_vector<real_type> > map(3);
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
 * @brief two-dimensional GridX with BasicCartesian metric
 */
template<class real_type>
struct BasicCartesianGridX2d: public dg::aBasicGeometryX2d<real_type>
{
    ///@copydoc GridX2d::GridX2d()
    BasicCartesianGridX2d( real_type x0, real_type x1, real_type y0, real_type y1, real_type fx, real_type fy, unsigned n, unsigned Nx, unsigned Ny, bc bcx = PER, bc bcy = PER): dg::aBasicGeometryX2d<real_type>(x0,x1,y0,y1,fx,fy,n,Nx,Ny,bcx,bcy){}
    /**
     * @brief Construct from existing topology
     * @param g existing grid class
     */
    BasicCartesianGridX2d( const dg::GridX2d& g):dg::aBasicGeometryX2d<real_type>(g.x0(),g.x1(),g.y0(),g.y1(),g.fx(),g.fy(),g.n(),g.Nx(),g.Ny(),g.bcx(),g.bcy()){}
    virtual BasicCartesianGridX2d* clone()const override final{
        return new BasicCartesianGridX2d(*this);
    }
};

/**
 * @brief three-dimensional GridX with BasicCartesian metric
 */
template<class real_type>
struct BasicCartesianGridX3d: public dg::aBasicGeometryX3d<real_type>
{
    ///@copydoc GridX3d::GridX3d()
    BasicCartesianGridX3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, real_type fx, real_type fy, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): dg::aBasicGeometryX3d<real_type>(x0,x1,y0,y1,z0,z1,fx,fy,n,Nx,Ny,Nz,bcx,bcy,bcz){}
    /**
     * @brief Implicit type conversion from GridX3d
     * @param g existing grid class
     */
    BasicCartesianGridX3d( const dg::GridX3d& g):dg::aBasicGeometryX3d<real_type>(g.x0(), g.x1(), g.y0(), g.y1(), g.z0(), g.z1(),g.fx(),g.fy(),g.n(),g.Nx(),g.Ny(),g.Nz(),g.bcx(),g.bcy(),g.bcz()){}
    virtual BasicCartesianGridX3d* clone()const override final{
        return new BasicCartesianGridX3d(*this);
    }
};

using CartesianGridX2d = BasicCartesianGridX2d<double>;
using CartesianGridX3d = BasicCartesianGridX3d<double>;
using aGeometryX2d = aBasicGeometryX2d<double>;
using aGeometryX3d = aBasicGeometryX3d<double>;

///@}

///@copydoc pullback(const Functor&,const aBasicGeometry2d&)
///@ingroup pullback
template< class Functor, class real_type>
thrust::host_vector<real_type> pullback( const Functor& f, const aBasicGeometryX2d<real_type>& g)
{
    std::vector<thrust::host_vector<real_type> > map = g.map();
    thrust::host_vector<real_type> vec( g.size());
    for( unsigned i=0; i<g.size(); i++)
        vec[i] = f( map[0][i], map[1][i]);
    return vec;
}

///@copydoc pullback(const Functor&,const aBasicGeometry2d&)
///@ingroup pullback
template< class Functor, class real_type>
thrust::host_vector<real_type> pullback( const Functor& f, const aBasicGeometryX3d<real_type>& g)
{
    std::vector<thrust::host_vector<real_type> > map = g.map();
    thrust::host_vector<real_type> vec( g.size());
    for( unsigned i=0; i<g.size(); i++)
        vec[i] = f( map[0][i], map[1][i], map[2][i]);
    return vec;
}
} //namespace dg
