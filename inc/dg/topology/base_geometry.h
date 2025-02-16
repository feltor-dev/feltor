#pragma once

#include "grid.h"
#include "tensor.h"

namespace dg
{

///@addtogroup basicgeometry
///@{

///@brief This is the abstract interface class for a two-dimensional Geometry
template<class real_type>
struct aRealGeometry2d : public aRealTopology<real_type,2>
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
    /// Inherit all parent constructors
    using aRealTopology<real_type,2>::aRealTopology;
    ///@copydoc aRealTopology::aRealTopology(const aRealTopology&)
    aRealGeometry2d( const aRealGeometry2d& src) = default;
    ///@copydoc aRealTopology::operator=(const aRealTopology&)
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
struct aRealGeometry3d : public aRealTopology<real_type,3>
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
    /// Inherit all parent constructors
    using aRealTopology<real_type,3>::aRealTopology;
    ///@copydoc aRealTopology::aRealTopology(const aRealTopology&)
    aRealGeometry3d( const aRealGeometry3d& src) = default;
    ///@copydoc aRealTopology::operator=(const aRealTopology&)
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
     * @attention The user takes ownership of the newly allocated grid
     * @code
     * dg::ClonePtr<aRealGeometry2d<real_type>> perp_ptr = grid.perp_grid();
     * @endcode
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
    ///@copydoc aRealTopology::aRealTopology(const aRealTopology&)
    aRealProductGeometry3d( const aRealProductGeometry3d& src) = default;
    ///@copydoc aRealTopology::operator=(const aRealTopology&)
    aRealProductGeometry3d& operator=( const aRealProductGeometry3d& src) = default;
    private:
    virtual aRealGeometry2d<real_type>* do_perp_grid()const=0;
};

/**
 * @brief Two-dimensional Grid with Cartesian metric

 * @snippet arakawa_t.cpp doxygen
 */
template<class real_type>
struct RealCartesianGrid2d: public dg::aRealGeometry2d<real_type>
{
    ///@copydoc RealGrid::RealGrid()
    RealCartesianGrid2d() = default;
    ///@copydoc hide_grid_parameters2d
    ///@copydoc hide_bc_parameters2d
    RealCartesianGrid2d( real_type x0, real_type x1, real_type y0, real_type y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx = PER, bc bcy = PER): dg::aRealGeometry2d<real_type>({x0,y0},{x1,y1},{n,n},{Nx,Ny},{bcx,bcy}){}

    /**
     * @brief Construct from given 1d grids
     * @param gx Axis 0 grid
     * @param gy Axis 1 grid
     */
    RealCartesianGrid2d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy):
        dg::aRealGeometry2d<real_type>(std::array{gx,gy}){}
    /**
     * @brief Construct from existing 2d topology
     * @param g existing grid class
     */
    RealCartesianGrid2d( const dg::RealGrid2d<real_type>& g):
        dg::aRealGeometry2d<real_type>(std::array{g.gx(), g.gy()}){}
    /// Enable ClonePtr
    virtual RealCartesianGrid2d<real_type>* clone()const override final{
        return new RealCartesianGrid2d<real_type>(*this);
    }
    private:
    virtual void do_set(std::array<unsigned,2> new_n, std::array<unsigned,2> new_N) override final{
        aRealTopology<real_type,2>::do_set(new_n,new_N);
    }
    virtual void do_set_pq( std::array<real_type,2> new_x0, std::array<real_type,2> new_x1) override final{
        aRealTopology<real_type,2>::do_set_pq(new_x0,new_x1);
    }
    virtual void do_set( std::array<dg::bc,2> new_bcs) override final{
        aRealTopology<real_type,2>::do_set(new_bcs);
    }
};

/**
 * @brief Three-dimensional Grid with Cartesian metric
 */
template<class real_type>
struct RealCartesianGrid3d: public dg::aRealProductGeometry3d<real_type>
{
    using perpendicular_grid = RealCartesianGrid2d<real_type>;
    RealCartesianGrid3d() = default;
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters3d
    RealCartesianGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1,
    unsigned n, unsigned Nx, unsigned Ny, unsigned Nz,
    bc bcx = PER, bc bcy = PER, bc bcz = PER):
        dg::aRealProductGeometry3d<real_type>({x0,y0,z0},{x1,y1,z1},{n,n,1},{Nx,Ny,Nz},{bcx,bcy,bcz})
        {}

    /**
     * @brief Construct from given 1d grids
     * @param gx Axis 0 grid
     * @param gy Axis 1 grid
     * @param gz Axis 2 grid
     */
    RealCartesianGrid3d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy, RealGrid1d<real_type> gz):
        dg::aRealProductGeometry3d<real_type>(std::array{gx,gy,gz}){}
    /**
     * @brief Implicit type conversion from Grid3d
     * @param g existing grid object
     */
    RealCartesianGrid3d( const dg::RealGrid3d<real_type>& g):
        dg::aRealProductGeometry3d<real_type>(std::array{g.gx(), g.gy(), g.gz()}){}
    /// Enable ClonePtr
    virtual RealCartesianGrid3d* clone()const override final{
        return new RealCartesianGrid3d(*this);
    }
    private:
    virtual RealCartesianGrid2d<real_type>* do_perp_grid() const override final{
        return new RealCartesianGrid2d<real_type>(this->gx(), this->gy());
    }
    virtual void do_set(std::array<unsigned,3> new_n, std::array<unsigned,3> new_N) override final{
        aRealTopology<real_type,3>::do_set(new_n,new_N);
    }
    virtual void do_set_pq( std::array<real_type,3> new_x0, std::array<real_type,3> new_x1) override final{
        aRealTopology<real_type,3>::do_set_pq(new_x0,new_x1);
    }
    virtual void do_set( std::array<dg::bc,3> new_bcs) override final{
        aRealTopology<real_type,3>::do_set(new_bcs);
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
    RealCylindricalGrid3d() = default;
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters3d
    ///@note x corresponds to R, y to Z and z to phi, the volume element is R
    RealCylindricalGrid3d( real_type x0, real_type x1,
        real_type y0, real_type y1, real_type z0, real_type z1,
        unsigned n, unsigned Nx, unsigned Ny, unsigned Nz,
        bc bcx = PER, bc bcy = PER, bc bcz = PER):
    dg::aRealProductGeometry3d<real_type>({x0,y0,z0},{x1,y1,z1},{n,n,1},{Nx,Ny,Nz},{bcx,bcy,bcz})
    {}
    /**
     * @brief Construct from given 1d grids
     * @param gx Axis 0 grid ->R
     * @param gy Axis 1 grid ->Z
     * @param gz Axis 2 grid ->P
     */
    RealCylindricalGrid3d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy, RealGrid1d<real_type> gz):
        dg::aRealProductGeometry3d<real_type>(std::array{gx,gy,gz}){}
    ///Enable ClonePtr
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
    virtual void do_set(std::array<unsigned,3> new_n, std::array<unsigned,3> new_N) override final{
        aRealTopology3d<real_type>::do_set(new_n,new_N);
    }
    virtual void do_set_pq( std::array<real_type,3> new_x0, std::array<real_type,3> new_x1) override final{
        aRealTopology<real_type,3>::do_set_pq(new_x0,new_x1);
    }
    virtual void do_set( std::array<dg::bc,3> new_bcs) override final{
        aRealTopology<real_type,3>::do_set(new_bcs);
    }
};

///@}

//TODO With C++17 class type inference we could rethink these?
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
