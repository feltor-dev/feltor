#pragma once

#include "grid.h"
#include "tensor.h"

namespace dg
{

///@addtogroup basicgeometry
///@{

///@brief This is the abstract interface class for a two-dimensional Geometry
template<class real_type>
struct aBasicGeometry2d : public aBasicTopology2d<real_type>
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
    */
    std::vector<thrust::host_vector<real_type> > map()const{
        return do_compute_map();
    }
    ///Geometries are cloneable
    virtual aBasicGeometry2d* clone()const=0;
    ///allow deletion through base class pointer
    virtual ~aBasicGeometry2d(){}
    protected:
    using aBasicTopology2d<real_type>::aBasicTopology2d;
    ////*!
    // * @copydoc aTopology2d::aTopology2d()
    // * @note the default coordinate map will be the identity
    // */
    //aBasicGeometry2d( real_type x0, real_type x1, real_type y0, real_type y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy):aTopology2d( x0,x1,y0,y1,n,Nx,Ny,bcx,bcy){}
    ///@copydoc aBasicTopology2d::aBasicTopology2d(const aBasicTopology2d&)
    aBasicGeometry2d( const aBasicGeometry2d& src):aBasicTopology2d<real_type>(src){}
    ///@copydoc aBasicTopology2d::operator=(const aBasicTopology2d&)
    aBasicGeometry2d& operator=( const aBasicGeometry2d& src){
        aBasicTopology2d<real_type>::operator=(src);
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

///@brief This is the abstract interface class for a three-dimensional Geometry
template<class real_type>
struct aBasicGeometry3d : public aBasicTopology3d<real_type>
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
    SparseTensor<thrust::host_vector<real_type> > jacobian()const{
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
    * @note use the dg::tensor functions to compute the volume element from here
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
    */
    std::vector<thrust::host_vector<real_type> > map()const{
        return do_compute_map();
    }
    ///Geometries are cloneable
    virtual aBasicGeometry3d* clone()const=0;
    ///allow deletion through base class pointer
    virtual ~aBasicGeometry3d(){}
    protected:
    using aBasicTopology3d<real_type>::aBasicTopology3d;
    ////*!
    // * @copydoc aBasicTopology3d::aBasicTopology3d()
    // * @note the default coordinate map will be the identity
    // */
    //aBasicGeometry3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz): aBasicTopology3d(x0,x1,y0,y1,z0,z1,n,Nx,Ny,Nz,bcx,bcy,bcz){}
    ///@copydoc aBasicTopology3d::aBasicTopology3d(const aBasicTopology3d&)
    aBasicGeometry3d( const aBasicGeometry3d& src):aBasicTopology3d<real_type>(src){}
    ///@copydoc aBasicTopology3d::operator=(const aBasicTopology3d&)
    aBasicGeometry3d& operator=( const aBasicGeometry3d& src){
        aBasicTopology3d<real_type>::operator=(src);
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

///@brief a 3d product space Geometry
template<class real_type>
struct aBasicProductGeometry3d : public aBasicGeometry3d<real_type>
{
    /*!
     * @brief The grid made up by the first two dimensions
     *
     * This is possible because the 3d grid is a product grid of a 2d perpendicular grid and a 1d parallel grid
     * @return A newly constructed perpendicular grid
     */
    aBasicGeometry2d<real_type>* perp_grid()const{
        return do_perp_grid();
    }
    ///allow deletion through base class pointer
    virtual ~aBasicProductGeometry3d(){}
    ///Geometries are cloneable
    virtual aBasicProductGeometry3d* clone()const=0;
    protected:
    ///@copydoc aBasicTopology3d::aBasicTopology3d(const aBasicTopology3d&)
    aBasicProductGeometry3d( const aBasicProductGeometry3d& src):aBasicGeometry3d<real_type>(src){}
    ///@copydoc aBasicTopology3d::operator=(const aBasicTopology3d&)
    aBasicProductGeometry3d& operator=( const aBasicProductGeometry3d& src){
        aBasicGeometry3d<real_type>::operator=(src);
        return *this;
    }
    /*!
     * @copydoc aBasicTopology3d::aBasicTopology3d()
     * @note the default coordinate map will be the identity
     */
    aBasicProductGeometry3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz): aBasicGeometry3d<real_type>(x0,x1,y0,y1,z0,z1,n,Nx,Ny,Nz,bcx,bcy,bcz){}
    private:
    virtual aBasicGeometry2d<real_type>* do_perp_grid()const=0;
};

using aGeometry2d = aBasicGeometry2d<double>;
using aGeometry3d = aBasicGeometry3d<double>;
using aProductGeometry3d = aBasicProductGeometry3d<double>;
///@}

///@addtogroup geometry
///@{

/**
 * @brief two-dimensional Grid with Cartesian metric

 * @snippet arakawa_t.cu doxygen
 */
template<class real_type>
struct BasicCartesianGrid2d: public dg::aBasicGeometry2d<real_type>
{
    ///@copydoc Grid2d::Grid2d()
    BasicCartesianGrid2d( real_type x0, real_type x1, real_type y0, real_type y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx = PER, bc bcy = PER): dg::aBasicGeometry2d<real_type>(x0,x1,y0,y1,n,Nx,Ny,bcx,bcy){}
    /**
     * @brief Construct from existing topology
     * @param g existing grid class
     */
    BasicCartesianGrid2d( const dg::BasicGrid2d<real_type>& g):dg::aBasicGeometry2d<real_type>(g.x0(),g.x1(),g.y0(),g.y1(),g.n(),g.Nx(),g.Ny(),g.bcx(),g.bcy()){}
    virtual BasicCartesianGrid2d<real_type>* clone()const override final{
        return new BasicCartesianGrid2d<real_type>(*this);
    }
    private:
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny) override final{
        aBasicTopology2d<real_type>::do_set(new_n,new_Nx,new_Ny);
    }
};

/**
 * @brief three-dimensional Grid with Cartesian metric
 */
template<class real_type>
struct BasicCartesianGrid3d: public dg::aBasicProductGeometry3d<real_type>
{
    using perpendicular_grid = BasicCartesianGrid2d<real_type>;
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters3d
    BasicCartesianGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): dg::aBasicProductGeometry3d<real_type>(x0,x1,y0,y1,z0,z1,n,Nx,Ny,Nz,bcx,bcy,bcz){}
    /**
     * @brief Implicit type conversion from Grid3d
     * @param g existing grid object
     */
    BasicCartesianGrid3d( const dg::BasicGrid3d<real_type>& g):dg::aBasicProductGeometry3d<real_type>(g.x0(), g.x1(), g.y0(), g.y1(), g.z0(), g.z1(),g.n(),g.Nx(),g.Ny(),g.Nz(),g.bcx(),g.bcy(),g.bcz()){}
    virtual BasicCartesianGrid3d* clone()const override final{
        return new BasicCartesianGrid3d(*this);
    }
    private:
    virtual BasicCartesianGrid2d<real_type>* do_perp_grid() const override final{
        return new BasicCartesianGrid2d<real_type>(this->x0(),this->x1(),this->y0(),this->y1(),this->n(),this->Nx(),this->Ny(),this->bcx(),this->bcy());
    }
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz) override final{
        aBasicTopology3d<real_type>::do_set(new_n,new_Nx,new_Ny,new_Nz);
    }
};

/**
 * @brief three-dimensional Grid with Cylindrical metric
 */
template<class real_type>
struct BasicCylindricalGrid3d: public dg::aBasicProductGeometry3d<real_type>
{
    using perpendicular_grid = BasicCartesianGrid2d<real_type>;
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters3d
    ///@note x corresponds to R, y to Z and z to phi, the volume element is R
    BasicCylindricalGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER): dg::aBasicProductGeometry3d<real_type>(x0,x1,y0,y1,z0,z1,n,Nx,Ny,Nz,bcx,bcy,bcz){}
    virtual BasicCylindricalGrid3d* clone()const override final{
        return new BasicCylindricalGrid3d(*this);
    }
    private:
    virtual BasicCartesianGrid2d<real_type>* do_perp_grid() const override final{
        return new BasicCartesianGrid2d<real_type>(this->x0(),this->x1(),this->y0(),this->y1(),this->n(),this->Nx(),this->Ny(),this->bcx(),this->bcy());
    }
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_metric()const override final{
        SparseTensor<thrust::host_vector<real_type> > metric(1);
        thrust::host_vector<real_type> R = dg::evaluate(dg::cooX3d, *this);
        for( unsigned i = 0; i<this->size(); i++)
            R[i] = 1./R[i]/R[i];
        metric.idx(2,2)=0;
        metric.values()[0] = R;
        return metric;
    }
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz) override final {
        aBasicTopology3d<real_type>::do_set(new_n,new_Nx,new_Ny,new_Nz);
    }
};

using CartesianGrid2d = BasicCartesianGrid2d<double>;
using CartesianGrid3d = BasicCartesianGrid3d<double>;
using CylindricalGrid3d = BasicCylindricalGrid3d<double>;
///@}

} //namespace dg
