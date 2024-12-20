#pragma once

#include "prolongation.h"
#ifdef MPI_VERSION
#include "mpi_prolongation.h"
#endif
#include <cusp/print.h>

/*! @file
  @brief Classes for poloidal and toroidal average computations.
  */
namespace dg{

/*!@brief Transpose vector

The equivalent of
    <tt> out[j*ny+i] = in[i*nx+j];</tt>
 * @copydoc hide_ContainerType
 * @param nx number of columns in input vector (size of contiguous chunks) /rows in output vector
 * @param ny number of rows in input vector /columns in output vector
 * @param in input
 * @param out output (may not alias in)
 * @attention Implemented using \c dg::blas2::parallel_for so **no MPI**, **no Recursive**
 * @ingroup utilities
*/
template<class ContainerType>
void transpose( unsigned nx, unsigned ny, const ContainerType& in, ContainerType& out)
{
    assert(&in != &out);
    using value_type = get_value_type<ContainerType>;
    dg::blas2::parallel_for( [nx,ny]DG_DEVICE( unsigned k, const value_type* ii, value_type* oo)
        {
            unsigned i = k/nx, j =  k%nx;
            oo[j*ny+i] = ii[i*nx+j];
        }, nx*ny, in, out);
}
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
    // TODO update docu
    template<class Topology, size_t Md>
    Average( const Topology& g, std::array<unsigned, Md> axes)
    {
        // Require restrictions on axes
        m_prolongation = dg::create::prolongation( g, axes);
        m_average      = dg::create::projection( axes, g);
        //cusp::print( m_average);

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
    /**
     * @brief Prepare internal workspace
     *
     * @param g the grid from which to take the dimensionality and sizes
     * @param direction the direction over which to average when calling \c operator()
     */
    template< class Topology, typename = std::enable_if_t<Topology::ndim() == 2>>
    Average( const Topology& g, enum coo2d direction) : Average( g, coo2axis(direction)) { }

    ///@copydoc Average()
    template< class Topology, typename = std::enable_if_t<Topology::ndim() == 3>>
    Average( const Topology& g, enum coo3d direction) : Average( g, coo2axis(direction)) { }

    template< class Topology, typename = std::enable_if_t<Topology::ndim() == 3>>
    Average( const Topology& g, std::array<enum coo3d,2> directions) :
        Average( g, {coo2axis(directions[0]), coo2axis(directions[1])}) { }
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
