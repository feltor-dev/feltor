#pragma once

#include "../backend/grid.h"
#include "tensor.h"

namespace dg
{

///@addtogroup basicgeometry
///@{

/**
 * @brief This is the abstract interface class for a two-dimensional Geometry
 */
struct aGeometry2d : public aTopology2d
{
    /**
    * @brief The Jacobian of the coordinate transformation from physical to computational space
    *
    *  The elements of the Tensor are (if x,y are the coordinates in computational space and R,Z are the physical space coordinates)
    \f[
    J = \begin{pmatrix} x_R(x,y) & x_Z(x,y) \\ y_R(x,y) & y_Z(x,y) 
    \end{pmatrix}
    \f]
    * @return Jacobian 
    */
    SparseTensor<thrust::host_vector<double> > jacobian()const{
        return do_compute_jacobian();
    }
    /**
    * @brief The Metric tensor of the coordinate system 
    *
    *  The elements are the contravariant elements (if x,y are the coordinates) 
    \f[
    g = \begin{pmatrix} g^{xx}(x,y) & g^{xy}(x,y) \\  & g^{yy}(x,y) \end{pmatrix}
    \f]
    * @return symmetric tensor
    */
    SparseTensor<thrust::host_vector<double> > metric()const { 
        return do_compute_metric(); 
    }
    /**
    * @brief The coordinate map from computational to physical space
    *
    *  The elements of the map are (if x,y are the coordinates in computational space and R,Z are the physical space coordinates)
    \f[
    R(x,y) \\
    Z(x,y)
    \f]
    * @return a vector of size 2 
    */
    std::vector<thrust::host_vector<double> > map()const{
        return do_compute_map();
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
    aGeometry2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy):aTopology2d( x0,x1,y0,y1,n,Nx,Ny,bcx,bcy){}
    ///@copydoc aTopology2d::aTopology2d(const aTopology2d&)
    aGeometry2d( const aGeometry2d& src):aTopology2d(src){}
    ///@copydoc aTopology2d::operator=(const aTopology2d&)
    aGeometry2d& operator=( const aGeometry2d& src){
        aTopology2d::operator=(src);
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
 * @brief This is the abstract interface class for a three-dimensional Geometry
 */
struct aGeometry3d : public aTopology3d
{
    /**
    * @brief The Jacobian of the coordinate transformation from physical to computational space
    *
    *  The elements of the Tensor are (if x,y,z are the coordinates in computational space and R,Z,P are the physical space coordinates)
    \f[
    J = \begin{pmatrix} x_R(x,y,z) & x_Z(x,y,z) & x_\varphi(x,y,z) \\ 
    y_R(x,y,z) & y_Z(x,y,z) & y_\varphi(x,y,z) \\
    z_R(x,y,z) & z_Z(x,y,z) & z_\varphi(x,y,z)
    \end{pmatrix}
    \f]
    * @return Jacobian 
    */
    SparseTensor<thrust::host_vector<double> > jacobian()const{
        return do_compute_jacobian();
    }
    /**
    * @brief The Metric tensor of the coordinate system 
    *
    *  The elements are the contravariant elements (if x,y,z are the coordinates) 
    \f[
    g = \begin{pmatrix} g^{xx}(x,y,z) & g^{xy}(x,y,z) & g^{zz}(x,y,z)\\
      & g^{yy}(x,y,z) & g^{yz}(x,y,z) \\
      & & g^{zz}(x,y,z)\end{pmatrix}
    \f]
    * @return symmetric tensor
    */
    SparseTensor<thrust::host_vector<double> > metric()const { 
        return do_compute_metric(); 
    }
    /**
    * @brief The coordinate map from computational to physical space
    *
    *  The elements of the map are (if x,y,z are the coordinates in computational space and R,Z,P are the physical space coordinates)
    \f[
    R(x,y,z) \\
    Z(x,y,z) \\
    \varphi(x,y,z)
    \f]
    * @return a vector of size 3 
    */
    std::vector<thrust::host_vector<double> > map()const{
        return do_compute_map();
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
    ///@copydoc aTopology3d::aTopology3d(const aTopology3d&)
    aGeometry3d( const aGeometry3d& src):aTopology3d(src){}
    ///@copydoc aTopology3d::operator=(const aTopology3d&)
    aGeometry3d& operator=( const aGeometry3d& src){
        aTopology3d::operator=(src);
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
 * @brief two-dimensional Grid with Cartesian metric
 */
struct CartesianGrid2d: public dg::aGeometry2d
{
    ///@copydoc Grid2d::Grid2d()
    CartesianGrid2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx = PER, bc bcy = PER): dg::aGeometry2d(x0,x1,y0,y1,n,Nx,Ny,bcx,bcy){}
    /**
     * @brief Construct from existing topology
     * @param g existing grid class
     */
    CartesianGrid2d( const dg::Grid2d& g):dg::aGeometry2d(g.x0(),g.x1(),g.y0(),g.y1(),g.n(),g.Nx(),g.Ny(),g.bcx(),g.bcy()){}
    virtual CartesianGrid2d* clone()const{return new CartesianGrid2d(*this);}
    private:
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny){
        aTopology2d::do_set(new_n,new_Nx,new_Ny);
    }
};

/**
 * @brief three-dimensional Grid with Cartesian metric
 */
struct CartesianGrid3d: public dg::aGeometry3d
{
    typedef CartesianGrid2d perpendicular_grid;
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters3d
    CartesianGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): dg::aGeometry3d(x0,x1,y0,y1,z0,z1,n,Nx,Ny,Nz,bcx,bcy,bcz){}
    /**
     * @brief Implicit type conversion from Grid3d
     * @param g existing grid object
     */
    CartesianGrid3d( const dg::Grid3d& g):dg::aGeometry3d(g.x0(), g.x1(), g.y0(), g.y1(), g.z0(), g.z1(),g.n(),g.Nx(),g.Ny(),g.Nz(),g.bcx(),g.bcy(),g.bcz()){}
    virtual CartesianGrid3d* clone()const{return new CartesianGrid3d(*this);}
    /*!
     * @brief The grid made up by the first two dimensions
     *
     * This is possible because the 3d grid is a product grid of a 2d perpendicular grid and a 1d parallel grid
     * @return A newly constructed perpendicular grid
     */
    CartesianGrid2d perp_grid() const{ return CartesianGrid2d(x0(),x1(),y0(),y1(),n(),Nx(),Ny(),bcx(),bcy());}
    private:
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz){
        aTopology3d::do_set(new_n,new_Nx,new_Ny,new_Nz);
    }
};

/**
 * @brief three-dimensional Grid with Cylindrical metric
 */
struct CylindricalGrid3d: public dg::aGeometry3d
{
    typedef CartesianGrid2d perpendicular_grid;
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters3d
    ///@note x corresponds to R, y to Z and z to phi, the volume element is R
    CylindricalGrid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): dg::aGeometry3d(x0,x1,y0,y1,z0,z1,n,Nx,Ny,Nz,bcx,bcy,bcz){}
    virtual CylindricalGrid3d* clone()const{return new CylindricalGrid3d(*this);}
    ///@copydoc  CartesianGrid3d::perp_grid()const
    CartesianGrid2d perp_grid() const{ return CartesianGrid2d(x0(),x1(),y0(),y1(),n(),Nx(),Ny(),bcx(),bcy());}
    private:
    virtual SparseTensor<thrust::host_vector<double> > do_compute_metric()const{
        SparseTensor<thrust::host_vector<double> > metric(1);
        thrust::host_vector<double> R = dg::evaluate(dg::cooX3d, *this);
        for( unsigned i = 0; i<size(); i++)
            R[i] = 1./R[i]/R[i];
        metric.idx(2,2)=0;
        metric.value(0) = R;
        return metric;
    }
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz){
        aTopology3d::do_set(new_n,new_Nx,new_Ny,new_Nz);
    }
};

///@}

} //namespace dg
