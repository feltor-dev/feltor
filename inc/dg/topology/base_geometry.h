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
    * @brief The Metric tensor of the coordinate system
    *
    *  The elements are the contravariant elements (if x,y are the coordinates)
    \f[
    g = \begin{pmatrix} g^{xx}(x,y) & g^{xy}(x,y) \\  & g^{yy}(x,y) \end{pmatrix}
    \f]
    * @return symmetric tensor
    * @note use the dg::tensor functions to compute the volume element from here
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
    * @brief The (contravariant) metric tensor of the coordinate system
    *
    *  The elements are the contravariant elements (if x,y,z are the coordinates)
    \f[
    g = \begin{pmatrix} g^{xx}(x,y,z) & g^{xy}(x,y,z) & g^{zz}(x,y,z)\\
      & g^{yy}(x,y,z) & g^{yz}(x,y,z) \\
      & & g^{zz}(x,y,z)\end{pmatrix}
    \f]
    * @return symmetric tensor
    * @note use the dg::tensor functions to compute the volume element from here
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

///@brief a 3d product space Geometry
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
    RealCartesianGrid2d( real_type x0, real_type x1, real_type y0, real_type y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx = PER, bc bcy = PER): dg::aRealGeometry2d<real_type>(x0,x1,y0,y1,n,Nx,Ny,bcx,bcy){}
    /**
     * @brief Construct from existing topology
     * @param g existing grid class
     */
    RealCartesianGrid2d( const dg::RealGrid2d<real_type>& g):dg::aRealGeometry2d<real_type>(g.x0(),g.x1(),g.y0(),g.y1(),g.n(),g.Nx(),g.Ny(),g.bcx(),g.bcy()){}
    virtual RealCartesianGrid2d<real_type>* clone()const override final{
        return new RealCartesianGrid2d<real_type>(*this);
    }
    private:
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny) override final{
        aRealTopology2d<real_type>::do_set(new_n,new_Nx,new_Ny);
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
    RealCartesianGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): dg::aRealProductGeometry3d<real_type>(x0,x1,y0,y1,z0,z1,n,Nx,Ny,Nz,bcx,bcy,bcz){}
    /**
     * @brief Implicit type conversion from Grid3d
     * @param g existing grid object
     */
    RealCartesianGrid3d( const dg::RealGrid3d<real_type>& g):dg::aRealProductGeometry3d<real_type>(g.x0(), g.x1(), g.y0(), g.y1(), g.z0(), g.z1(),g.n(),g.Nx(),g.Ny(),g.Nz(),g.bcx(),g.bcy(),g.bcz()){}
    virtual RealCartesianGrid3d* clone()const override final{
        return new RealCartesianGrid3d(*this);
    }
    private:
    virtual RealCartesianGrid2d<real_type>* do_perp_grid() const override final{
        return new RealCartesianGrid2d<real_type>(this->x0(),this->x1(),this->y0(),this->y1(),this->n(),this->Nx(),this->Ny(),this->bcx(),this->bcy());
    }
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz) override final{
        aRealTopology3d<real_type>::do_set(new_n,new_Nx,new_Ny,new_Nz);
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
    RealCylindricalGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): dg::aRealProductGeometry3d<real_type>(x0,x1,y0,y1,z0,z1,n,Nx,Ny,Nz,bcx,bcy,bcz){}
    virtual RealCylindricalGrid3d* clone()const override final{
        return new RealCylindricalGrid3d(*this);
    }
    private:
    virtual RealCartesianGrid2d<real_type>* do_perp_grid() const override final{
        return new RealCartesianGrid2d<real_type>(this->x0(),this->x1(),this->y0(),this->y1(),this->n(),this->Nx(),this->Ny(),this->bcx(),this->bcy());
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
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz) override final {
        aRealTopology3d<real_type>::do_set(new_n,new_Nx,new_Ny,new_Nz);
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
using aGeometry2d_t           = aGeometry2d           ;
using aGeometry3d_t           = aGeometry3d           ;
using aProductGeometry3d_t    = aProductGeometry3d    ;
using CartesianGrid2d_t       = CartesianGrid2d       ;
using CartesianGrid3d_t       = CartesianGrid3d       ;
using CylindricalGrid3d_t     = CylindricalGrid3d     ;
#endif //MPI_VERSION

///@}

} //namespace dg
