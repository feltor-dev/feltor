#pragma once

#include "prolongation.h"
#ifdef MPI_VERSION
#include "mpi_prolongation.h"
#endif

/*! @file
  @brief Classes for poloidal and toroidal average computations.
  */
namespace dg{

/**
 * @brief Topological average computations in a Cartesian topology
 *
 * \f{align}{
 * \langle f \rangle_x := \frac{1}{L_x}\int_0^{L_x}dx f \quad
 * \langle f \rangle_y := \frac{1}{L_y}\int_0^{L_y}dy f \quad
 * \langle f \rangle_z := \frac{1}{L_z}\int_0^{L_z}dz f \\
 * \langle f \rangle_{xy} := \frac{1}{L_xL_y}\int_0^{L_x}\int_0^{L_y}dxdy f \quad
 * \langle f \rangle_{xz} := \frac{1}{L_xL_z}\int_0^{L_x}\int_0^{L_z}dxdz f \quad
 * \langle f \rangle_{yz} := \frac{1}{L_yL_z}\int_0^{L_y}\int_0^{L_z}dydz f \quad
 * \f}
 * Given a Cartesian topology it is possible to define a partial reduction of a
 * given vector.  In two dimensions for example we can define a reduction over
 * all points that are neighbors in the x (or y) direction.  We are then left
 * with Ny (Nx) points. In three dimensions we can define the reduction along
 * the x, y, z directions but also over all points in the xy (xz or yz) planes.
 * We are left with two- (respectively three-)dimensional vectors.
 *
 * @sa Average is essentially a \c dg::create::projection(std::array<unsigned,Md>,const aRealTopology<real_type,Nd>&) scaled with the
 * inverse area @note The integrals include the dG weights but not the volume
 * element (does
 * not know about geometry)
 *
 * @snippet topology/average_t.cpp doxygen
 * @ingroup average
 */
template<class IMatrix, class ContainerType>
struct Average
{
    using container_type = ContainerType;
    using value_type = dg::get_value_type<ContainerType>;
    /// No allocation
    Average() = default;

    /*!@brief Average along given axes
     *
     * @tparam Md Number of dimensions to reduce \c Md<Topology::ndim()
     * @param axes Axis numbers in \c g_old along which to reduce
     * <tt>axes[i]<g_old.ndim()</tt>. Can be a named dimension from \c
     * dg::coo2d or \c dg::coo3d
     * @param g Grid of the old, un-reduced vectors
     * @sa dg::create::projection
     */
    template<class Topology, size_t Md>
    Average( const Topology& g, std::array<unsigned, Md> axes)
    {
        m_prolongation = dg::create::prolongation( g, axes);
        m_average      = dg::create::projection( axes, g);

        auto remains = dg::create::detail::complement( g, axes);

        auto tmp = dg::evaluate( dg::one, g.grid(remains[0]));
        for( unsigned u=1; u<remains.size(); u++)
            tmp = dg::kronecker( dg::Product(), tmp,
                dg::evaluate(dg::one, g.grid(remains[u])));  //note tmp comes first (important for comms in MPI)
        m_tmp = dg::construct<ContainerType>(tmp);
        m_area_inv = 1.;
        for( unsigned u=0; u<Md; u++)
            m_area_inv /= g.l(axes[u]);
    }

    ///@copydoc Average(const Topology&,std::array<unsigned,Md>)
    template< class Topology, typename = std::enable_if_t<Topology::ndim() == 2>>
    Average( const Topology& g, enum coo2d axes)
    : Average( g, coo2axis(axes))
    {
    }

    ///@copydoc Average(const Topology&,std::array<unsigned,Md>)
    template< class Topology, typename = std::enable_if_t<Topology::ndim() == 3>>
    Average( const Topology& g, enum coo3d axes)
    : Average( g, coo2axis(axes))
    {
    }

    ///@copydoc Average(const Topology&,std::array<unsigned,Md>)
    template< class Topology, typename = std::enable_if_t<Topology::ndim() == 3>>
    Average( const Topology& g, std::array<enum coo3d,2> axes)
    : Average( g, {coo2axis(axes[0]), coo2axis(axes[1])})
    {
    }

    /**
     * @brief Compute the average as configured in the constructor
     *
     * The compuatation is divided in two steps:
     *
     *  -# average the input field over the direction or plane given in the
     *  constructor
     *  -# [optionally] extend the lower dimensional result back to the
     *  original dimensionality
     * .
     *
     * @param src Source Vector (must have the same size as the grid given in
     * the constructor)
     * @param res result Vector (if \c extend==true, \c res must have same size
     * as \c src vector, else it gets properly resized, may alias \c src)
     * @param extend if \c true the average is extended back to the original
     * dimensionality, if \c false, this step is skipped
     */
    void operator() (const ContainerType& src, ContainerType& res, bool extend
            = true)
    {
        dg::blas2::symv( m_average, src, m_tmp);
        dg::blas1::scal( m_tmp, m_area_inv);
        if( extend )
            dg::blas2::symv( m_prolongation, m_tmp, res);
        else
            res = m_tmp;
    }

  private:
    std::array<unsigned, 1> coo2axis( enum coo2d direction)
    {
        if( direction == coo2d::x)
            return {0};
        return {1};
    }
    std::array<unsigned, 1> coo2axis( enum coo3d direction)
    {
        if( direction == coo3d::x)
            return {0};
        if( direction == coo3d::y)
            return {1};
        return {2};
    }
    ContainerType m_tmp;
    IMatrix m_average, m_prolongation;
    double m_area_inv;
};

}//namespace dg
