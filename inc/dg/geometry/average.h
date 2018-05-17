#pragma once

#include "grid.h"
#include "weights.cuh"
#include "dg/blas1.h"
#include "dg/backend/average_dispatch.h"

/*! @file
  @brief contains classes for poloidal and toroidal average computations.
  */
namespace dg{

/**
 * @brief Class for average computations over a given direction or plane
 *
 * @snippet geometry/average_t.cu doxygen
 * @ingroup utilities
 */
template< class container>
struct Average
{
    /**
     * @brief Prepare internal workspace
     *
     * @param g the grid from which to take the dimensionality and sizes
     * @param direction the direction or plane over which to average when calling \c operator() (at the moment cannot be \c coo3d::xz or \c coo3d::y)
     */
    Average( const aTopology2d& g, enum coo2d direction)
    {
        m_nx = g.Nx()*g.n(), m_ny = g.Ny()*g.n();
        m_w=dg::transfer<container>(dg::create::weights(g, direction));
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
        m_w = dg::transfer<container>(dg::create::weights(g, direction));
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
    void operator() (const container& src, container& res, bool extend = true)
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
    container m_w, m_temp, m_temp1d;
    bool m_transpose;

};


}//namespace dg
