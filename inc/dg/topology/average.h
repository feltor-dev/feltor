#pragma once

#include "grid.h"
#include "weights.h"
#include "dg/blas1.h"
#include "dg/backend/average_dispatch.h"

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
template< class ContainerType>
struct Average
{
    using container_type = ContainerType;
    /**
     * @brief Prepare internal workspace
     *
     * @param g the grid from which to take the dimensionality and sizes
     * @param direction the direction or plane over which to average when calling \c operator() (at the moment cannot be \c coo3d::xz or \c coo3d::y)
     */
    Average( const aTopology2d& g, enum coo2d direction)
    {
        m_nx = g.Nx()*g.n(), m_ny = g.Ny()*g.n();
        m_w=dg::construct<ContainerType>(dg::create::weights(g, direction));
        m_temp1d = m_temp = m_w;
        m_transpose = false;
        if( direction == coo2d::x)
            dg::blas1::scal( m_w, 1./g.lx());
        else
        {
            m_transpose = true;
            dg::blas1::scal( m_temp, 1./g.ly());
            dg::transpose( m_nx, m_ny, m_temp, m_w);
        }
    }

    ///@copydoc Average()
    Average( const aTopology3d& g, enum coo3d direction)
    {
        m_w = dg::construct<ContainerType>(dg::create::weights(g, direction));
        m_temp1d = m_temp = m_w;
        m_transpose = false;
        unsigned nx = g.n()*g.Nx(), ny = g.n()*g.Ny(), nz = g.Nz();
        if( direction == coo3d::x) {
            dg::blas1::scal( m_w, 1./g.lx());
            m_nx = nx, m_ny = ny*nz;
        }
        else if( direction == coo3d::z) {
            m_transpose = true;
            dg::blas1::scal( m_temp, 1./g.lz());
            m_nx = nx*ny, m_ny = nz;
            dg::transpose( m_nx, m_ny, m_temp, m_w);
        }
        else if( direction == coo3d::xy) {
            dg::blas1::scal( m_w, 1./g.lx()/g.ly());
            m_nx = nx*ny, m_ny = nz;
        }
        else if( direction == coo3d::yz) {
            m_transpose = true;
            dg::blas1::scal( m_temp, 1./g.ly()/g.lz());
            m_nx = nx, m_ny = ny*nz;
            dg::transpose( m_nx, m_ny, m_temp, m_w);
        }
        else
            std::cerr << "Warning: this direction is not implemented\n";
    }
    /**
     * @brief Compute the average as configured in the constructor
     *
     * The compuatation is based on the exact, reproducible scalar product provided in the \c ::exblas library. It is divided in two steps
     *  - average the input field over the direction or plane given in the constructor
     *  - extend the lower dimensional result back to the original dimensionality
     *
     * @param src Source Vector (must have the same size as the grid given in the constructor)
     * @param res result Vector (if \c extend==true, \c res must have same size as \c src vector, else it gets properly resized, may alias \c src)
     * @param extend if \c true the average is extended back to the original dimensionality, if \c false, this step is skipped
     */
    void operator() (const ContainerType& src, ContainerType& res, bool extend = true)
    {
        if( !m_transpose)
        {
            dg::average( m_nx, m_ny, src, m_w, m_temp1d);
            if( extend )
                dg::extend_column( m_nx, m_ny, m_temp1d, res);
            else
                res = m_temp1d;
        }
        else
        {
            dg::transpose( m_nx, m_ny, src, m_temp);
            dg::average( m_ny, m_nx, m_temp, m_w, m_temp1d);
            if( extend )
                dg::extend_line( m_nx, m_ny, m_temp1d, res);
            else
                res = m_temp1d;
        }
    }

  private:
    unsigned m_nx, m_ny;
    ContainerType m_w, m_temp, m_temp1d;
    bool m_transpose;

};


}//namespace dg
