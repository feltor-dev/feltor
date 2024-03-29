#pragma once

#include "grid.h"
#include "tensor.h"

namespace dg
{

///@addtogroup basicgeometry
///@{

///@brief This is the abstract interface class for a two-dimensional Geometry
template<class real_type>
struct aRealGeometry2d : public aRealTopology2d<real_type>
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
    * @note per default this will be the identity tensor
    */
    SparseTensor<thrust::host_vector<real_type> > jacobian()const{
        return do_compute_jacobian();
    }
    /**
    * @brief The (inverse) metric tensor of the coordinate system
    *
    *  The elements of the inverse metric tensor are the contravariant elements
    *  of the metric \f$g\f$. If x,y are the coordinates, then
    \f[
    g^{-1} = \begin{pmatrix} g^{xx}(x,y) & g^{xy}(x,y) \\  & g^{yy}(x,y) \end{pmatrix}
    \f]
    * @return symmetric tensor
    * @note use the \c dg::tensor::volume2d function to compute the volume element from here
    * @note per default this will be the identity tensor
    */
    SparseTensor<thrust::host_vector<real_type> > metric()const {
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
    * @note per default this will be the identity map
    */
    std::vector<thrust::host_vector<real_type> > map()const{
        return do_compute_map();
    }
    ///Geometries are cloneable
    virtual aRealGeometry2d* clone()const=0;
    ///allow deletion through base class pointer
    virtual ~aRealGeometry2d() = default;
    protected:
    using aRealTopology2d<real_type>::aRealTopology2d;
    ///@copydoc aRealTopology2d::aRealTopology2d(const aRealTopology2d&)
    aRealGeometry2d( const aRealGeometry2d& src) = default;
    ///@copydoc aRealTopology2d::operator=(const aRealTopology2d&)
    aRealGeometry2d& operator=( const aRealGeometry2d& src) = default;
    private:
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_metric()const {
        return SparseTensor<thrust::host_vector<real_type> >(*this);
    }
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_jacobian()const {
        return SparseTensor<thrust::host_vector<real_type> >(*this);
    }
    virtual std::vector<thrust::host_vector<real_type> > do_compute_map()const{
        std::vector<thrust::host_vector<real_type> > map(2);
        map[0] = dg::evaluate(dg::cooX2d, *this);
        map[1] = dg::evaluate(dg::cooY2d, *this);
        return map;
    }


};

///@brief This is the abstract interface class for a three-dimensional Geometry
template<class real_type>
struct aRealGeometry3d : public aRealTopology3d<real_type>
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
    * @note per default this will be the identity tensor
    */
    SparseTensor<thrust::host_vector<real_type> > jacobian()const{
        return do_compute_jacobian();
    }
    /**
    * @brief The (inverse) metric tensor of the coordinate system
    *
    *  The elements of the inverse metric tensor are the contravariant elements
    *  of the metric \f$g\f$. If x,y,z are the coordinates, then
    \f[
    g^{-1} = \begin{pmatrix} g^{xx}(x,y,z) & g^{xy}(x,y,z) & g^{zz}(x,y,z)\\
      & g^{yy}(x,y,z) & g^{yz}(x,y,z) \\
      & & g^{zz}(x,y,z)\end{pmatrix}
    \f]
    * @return symmetric tensor
    * @note use the \c dg::tensor::volume function to compute the volume element from here
    * @note per default this will be the identity tensor
    */
    SparseTensor<thrust::host_vector<real_type> > metric()const {
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
    * @note per default this will be the identity map
    */
    std::vector<thrust::host_vector<real_type> > map()const{
        return do_compute_map();
    }
    ///Geometries are cloneable
    virtual aRealGeometry3d* clone()const=0;
    ///allow deletion through base class pointer
    virtual ~aRealGeometry3d() = default;
    protected:
    using aRealTopology3d<real_type>::aRealTopology3d;
    ///@copydoc aRealTopology3d::aRealTopology3d(const aRealTopology3d&)
    aRealGeometry3d( const aRealGeometry3d& src) = default;
    ///@copydoc aRealTopology3d::operator=(const aRealTopology3d&)
    aRealGeometry3d& operator=( const aRealGeometry3d& src) = default;
    private:
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_metric()const {
        return SparseTensor<thrust::host_vector<real_type> >(*this);
    }
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_jacobian()const {
        return SparseTensor<thrust::host_vector<real_type> >(*this);
    }
    virtual std::vector<thrust::host_vector<real_type> > do_compute_map()const{
        std::vector<thrust::host_vector<real_type> > map(3);
        map[0] = dg::evaluate(dg::cooX3d, *this);
        map[1] = dg::evaluate(dg::cooY3d, *this);
        map[2] = dg::evaluate(dg::cooZ3d, *this);
        return map;
    }
};

/**
 * @brief A 3d product space Geometry \f$ g_{2d} \otimes g_{1d}\f$
 *
 * This class represents a product space of a 2d grid (the "perp_grid") and a 1d
 * grid (the "parallel_grid").
 * The special feature of the product space is that the metric is simply
 * \f[ g = \begin{pmatrix}
 *  (g_{2d}(x,y)) & 0 \\
 *  0 & g_{1d}(x,y)
 * \end{pmatrix}
 * \f]
 * That is the metric elements do not depend on the third coordinate.
 * @tparam real_type The value type of the grid
 */
template<class real_type>
struct aRealProductGeometry3d : public aRealGeometry3d<real_type>
{
    /*!
     * @brief The grid made up by the first two dimensions
     *
     * This is possible because the 3d grid is a product grid of a 2d perpendicular grid and a 1d parallel grid
     * @return A newly constructed perpendicular grid
     */
    aRealGeometry2d<real_type>* perp_grid()const{
        return do_perp_grid();
    }
    ///allow deletion through base class pointer
    virtual ~aRealProductGeometry3d() = default;
    ///Geometries are cloneable
    virtual aRealProductGeometry3d* clone()const=0;
    protected:
    using aRealGeometry3d<real_type>::aRealGeometry3d;
    ///@copydoc aRealTopology3d::aRealTopology3d(const aRealTopology3d&)
    aRealProductGeometry3d( const aRealProductGeometry3d& src) = default;
    ///@copydoc aRealTopology3d::operator=(const aRealTopology3d&)
    aRealProductGeometry3d& operator=( const aRealProductGeometry3d& src) = default;
    private:
    virtual aRealGeometry2d<real_type>* do_perp_grid()const=0;
};
///@}

///@addtogroup geometry
///@{

/**
 * @brief two-dimensional Grid with Cartesian metric

 * @snippet arakawa_t.cu doxygen
 */
template<class real_type>
struct RealCartesianGrid2d: public dg::aRealGeometry2d<real_type>
{
    ///@copydoc RealGrid2d::RealGrid2d()
    RealCartesianGrid2d( real_type x0, real_type x1, real_type y0, real_type y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx = PER, bc bcy = PER): dg::aRealGeometry2d<real_type>({x0,x1,n,Nx,bcx},{y0,y1,n,Ny,bcy}){}

    ///@copydoc aRealTopology2d<real_type>::aRealTopology2d(RealGrid1d<real_type>,RealGrid1d<real_type>)
    RealCartesianGrid2d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy): dg::aRealGeometry2d<real_type>(gx,gy){}
    /**
     * @brief Construct from existing topology
     * @param g existing grid class
     */
    RealCartesianGrid2d( const dg::RealGrid2d<real_type>& g):dg::aRealGeometry2d<real_type>(g.gx(), g.gy()){}
    virtual RealCartesianGrid2d<real_type>* clone()const override final{
        return new RealCartesianGrid2d<real_type>(*this);
    }
    private:
    virtual void do_set(unsigned nx, unsigned Nx, unsigned ny, unsigned Ny) override final{
        aRealTopology2d<real_type>::do_set(nx,Nx,ny,Ny);
    }
};

/**
 * @brief three-dimensional Grid with Cartesian metric
 */
template<class real_type>
struct RealCartesianGrid3d: public dg::aRealProductGeometry3d<real_type>
{
    using perpendicular_grid = RealCartesianGrid2d<real_type>;
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters3d
    RealCartesianGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): dg::aRealProductGeometry3d<real_type>({x0,x1,n,Nx,bcx}, {y0,y1,n,Ny,bcy},{z0,z1,1,Nz,bcz}){}

    ///@copydoc aRealTopology3d::aRealTopology3d(RealGrid1d,RealGrid1d,RealGrid1d)
    RealCartesianGrid3d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy, RealGrid1d<real_type> gz): dg::aRealProductGeometry3d<real_type>(gx,gy,gz){}
    /**
     * @brief Implicit type conversion from Grid3d
     * @param g existing grid object
     */
    RealCartesianGrid3d( const dg::RealGrid3d<real_type>& g):dg::aRealProductGeometry3d<real_type>(g.gx(), g.gy(), g.gz()){}
    virtual RealCartesianGrid3d* clone()const override final{
        return new RealCartesianGrid3d(*this);
    }
    private:
    virtual RealCartesianGrid2d<real_type>* do_perp_grid() const override final{
        return new RealCartesianGrid2d<real_type>(this->gx(), this->gy());
    }
    virtual void do_set(unsigned nx, unsigned Nx, unsigned ny, unsigned Ny, unsigned nz, unsigned Nz) override final{
        aRealTopology3d<real_type>::do_set(nx,Nx,ny,Ny,nz,Nz);
    }
};

/**
 * @brief three-dimensional Grid with Cylindrical metric
 * @note \c map() returns the identity, i.e. Cylindrical coordinates count as physical coordinates. Evaluate the \c dg::cooRZP2X() functions to transform to a Cartesian coordinate system.
 */
template<class real_type>
struct RealCylindricalGrid3d: public dg::aRealProductGeometry3d<real_type>
{
    using perpendicular_grid = RealCartesianGrid2d<real_type>;
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters3d
    ///@note x corresponds to R, y to Z and z to phi, the volume element is R
    RealCylindricalGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): dg::aRealProductGeometry3d<real_type>({x0,x1,n,Nx,bcx},{y0,y1,n,Ny,bcy},{z0,z1,1,Nz,bcz}){}
    ///@copydoc aRealTopology3d::aRealTopology3d(RealGrid1d,RealGrid1d,RealGrid1d)
    RealCylindricalGrid3d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy, RealGrid1d<real_type> gz): dg::aRealProductGeometry3d<real_type>(gx,gy,gz){}
    virtual RealCylindricalGrid3d* clone()const override final{
        return new RealCylindricalGrid3d(*this);
    }
    private:
    virtual RealCartesianGrid2d<real_type>* do_perp_grid() const override final{
        return new RealCartesianGrid2d<real_type>(this->gx(), this->gy());
    }
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_metric()const override final{
        SparseTensor<thrust::host_vector<real_type> > metric(*this);
        thrust::host_vector<real_type> R = dg::evaluate(dg::cooX3d, *this);
        for( unsigned i = 0; i<this->size(); i++)
            R[i] = 1./R[i]/R[i];
        metric.idx(2,2)=2;
        metric.values().push_back( R);
        return metric;
    }
    virtual void do_set(unsigned nx, unsigned Nx, unsigned ny,unsigned Ny, unsigned nz,unsigned Nz) override final {
        aRealTopology3d<real_type>::do_set(nx,Nx,ny,Ny,nz,Nz);
    }
};

///@}

///@addtogroup gridtypes
///@{
using aGeometry2d           = dg::aRealGeometry2d<double>;
using aGeometry3d           = dg::aRealGeometry3d<double>;
using aProductGeometry3d    = dg::aRealProductGeometry3d<double>;
using CartesianGrid2d       = dg::RealCartesianGrid2d<double>;
using CartesianGrid3d       = dg::RealCartesianGrid3d<double>;
using CylindricalGrid3d     = dg::RealCylindricalGrid3d<double>;
#ifndef MPI_VERSION
namespace x{
using aGeometry2d           = aGeometry2d           ;
using aGeometry3d           = aGeometry3d           ;
using aProductGeometry3d    = aProductGeometry3d    ;
using CartesianGrid2d       = CartesianGrid2d       ;
using CartesianGrid3d       = CartesianGrid3d       ;
using CylindricalGrid3d     = CylindricalGrid3d     ;
}//namespace x
#endif //MPI_VERSION

///@}

} //namespace dg
