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
    SparseTensor<thrust::host_vector<double> > jacobian()const{
        return do_compute_jacobian();
    }
    SparseTensor<thrust::host_vector<double> > metric()const { 
        return do_compute_metric(); 
    }
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
    aGeometryX2d( const aGeometryX2d& src):aTopologyX2d(src){}
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
    SparseTensor<thrust::host_vector<double> > jacobian()const{
        return do_compute_jacobian();
    }
    SparseTensor<thrust::host_vector<double> > metric()const { 
        return do_compute_metric(); 
    }
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
    aGeometryX3d( const aGeometryX3d& src):aTopologyX3d(src){}
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
    private:
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny){
        aTopologyX2d::do_set(new_n,new_Nx,new_Ny);
    }
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
    private:
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz){
        aTopologyX3d::do_set(new_n,new_Nx,new_Ny,new_Nz);
    }
};

///@}

} //namespace dg
