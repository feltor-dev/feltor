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
 * \f[
 * \langle f \rangle_x := \frac{1}{L_x}\int_0^{L_x}dx f \quad
 * \langle f \rangle_y := \frac{1}{L_y}\int_0^{L_y}dy f \quad
 * \langle f \rangle_z := \frac{1}{L_z}\int_0^{L_z}dz f \\
 * \langle f \rangle_{xy} := \frac{1}{L_xL_y}\int_0^{L_x}\int_0^{L_y}dxdy f \quad
 * \langle f \rangle_{xz} := \frac{1}{L_xL_z}\int_0^{L_x}\int_0^{L_z}dxdz f \quad
 * \langle f \rangle_{yz} := \frac{1}{L_yL_z}\int_0^{L_y}\int_0^{L_z}dydz f \quad
 * \f]
 * Given a Cartesian topology it is possible to define a partial reduction of a given vector.
 * In two dimensions for example we can define a reduction over all points that are neighbors in the x (or y) direction.
 * We are then left with Ny (Nx) points. In three dimensions we can define the reduction along the x, y, z directions
 * but also over all points in the xy (xz or yz) planes. We are left with two- (respectively three-)dimensional vectors.
 * @note The integrals include the dG weights but not the volume element (does not know about geometry)
 * @snippet topology/average_t.cu doxygen
 * @ingroup utilities
 */
template<class IMatrix, class ContainerType>
struct Average
{
    using container_type = ContainerType;
    using value_type = dg::get_value_type<ContainerType>;
    Average() = default;
    template<class Topology>
    Average( const Topology& g, std::array<bool, Topology::ndim()> remains)
    {
        // Here, the challenge is that the dimension of the remaining grid is a runtime parameter
        // because remains needs to be parsed
        // The alternative would be std::array<unsigned, Md> map in the interface
        // but that requires restrictions on map
        m_prolongation = dg::create::prolongation( g, remains);
        m_average = dg::create::average( remains, g);
        unsigned first_idx = 0;
        for( unsigned u=0; u<Topology::ndim(); u++)
            if( remains[u] )
            {
                first_idx = u;
                break;
            }
        auto tmp = dg::evaluate( dg::one, g.grid(first_idx));
        for( unsigned u=first_idx+1; u<Topology::ndim(); u++)
            if( remains[u] )
                tmp = dg::kronecker( dg::Product(), tmp, dg::evaluate( dg::one, g.grid(u)));  //note tmp comes first
        m_tmp = dg::construct<ContainerType>(tmp);
        m_area_inv = 1.;
        for( unsigned u=0; u<Topology::ndim(); u++)
            if( !remains[u])
                m_area_inv /= g.l(u);
    }
    /**
     * @brief Prepare internal workspace
     *
     * @param g the grid from which to take the dimensionality and sizes
     * @param direction the direction or plane over which to average when calling \c operator() (at the moment cannot be \c coo3d::xz or \c coo3d::y)
     */
    template< class Topology, typename = std::enable_if_t<Topology::ndim() == 2>>
    Average( const Topology& g, enum coo2d direction) : Average( g, coo2remains(direction)) { }

    ///@copydoc Average()
    template< class Topology, typename = std::enable_if_t<Topology::ndim() == 3>>
    Average( const Topology& g, enum coo3d direction) : Average( g, coo2remains(direction)) { }
    /**
     * @brief Compute the average as configured in the constructor
     *
     * The compuatation is based on the exact, reproducible scalar product provided in the \c dg::exblas library. It is divided in two steps
     *  - average the input field over the direction or plane given in the constructor
     *  - extend the lower dimensional result back to the original dimensionality
     *
     * @param src Source Vector (must have the same size as the grid given in the constructor)
     * @param res result Vector (if \c extend==true, \c res must have same size as \c src vector, else it gets properly resized, may alias \c src)
     * @param extend if \c true the average is extended back to the original dimensionality, if \c false, this step is skipped
     */
    void operator() (const ContainerType& src, ContainerType& res, bool extend = true)
    {
        dg::blas2::symv( m_average, src, m_tmp);
        dg::blas1::scal( m_tmp, m_area_inv);
        if( extend )
            dg::blas2::symv( m_prolongation, m_tmp, res);
        else
            res = m_tmp;
    }

  private:
    std::array<bool, 2> coo2remains( enum coo2d direction)
    {
        if( direction == coo2d::x)
            return {0,1};
        return {1,0};
    }
    std::array<bool, 3> coo2remains( enum coo3d direction)
    {
        if( direction == coo3d::x)
            return {0,1,1};
        if( direction == coo3d::y)
            return {1,0,1};
        if( direction == coo3d::z)
            return {1,1,0};
        if( direction == coo3d::xy)
            return {0,0,1};
        if( direction == coo3d::yz)
            return {1,0,0};
        return {0,1,0};
    }
    ContainerType m_tmp;
    IMatrix m_average, m_prolongation;
    double m_area_inv;
};

}//namespace dg
