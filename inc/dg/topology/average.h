#pragma once

#include "grid.h"
#include "weights.h"
#include "dg/blas1.h"
#include "dg/backend/average_dispatch.h"
#include "dg/backend/view.h"

/*! @file
  @brief Classes for poloidal and toroidal average computations.
  */
namespace dg{
///@cond
template<class container>
void simple_average( unsigned nx, unsigned ny, const container& in0, const container& in1, container& out)
{
    const double* in0_ptr = thrust::raw_pointer_cast( in0.data());
    const double* in1_ptr = thrust::raw_pointer_cast( in1.data());
          double* out_ptr = thrust::raw_pointer_cast( out.data());
    dg::View<const container> in0_view( in0_ptr, nx), in1_view( in1_ptr, nx);
    dg::View<container> out_view( out_ptr, nx);
    dg::blas1::pointwiseDot( in0_view, in1_view, out_view);
    for( unsigned i=1; i<ny; i++)
    {
        in0_view.construct( in0_ptr+i*nx, nx);
        in1_view.construct( in1_ptr+i*nx, nx);
        dg::blas1::pointwiseDot( 1., in0_view, in1_view, 1, out_view);
    }
}
///@endcond

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
    Average() = default;
    /**
     * @brief Prepare internal workspace
     *
     * @param g the grid from which to take the dimensionality and sizes
     * @param direction the direction or plane over which to average when calling \c operator() (at the moment cannot be \c coo3d::xz or \c coo3d::y)
     * @param mode either "exact" ( uses the exact and reproducible dot product for the summation) or "simple" (uses inexact but much faster direct summation) use simple if you do not need the reproducibility
     * @note computing in "exact" mode is especially difficult if the averaged
     * direction is small compared to the remaining dimensions and for GPUs in
     * general, expect to gain a factor 10-1000 (no joke) from going to
     * "simple" mode in these cases
     */
    Average( const aTopology2d& g, enum coo2d direction, std::string mode = "exact") : m_mode(mode)
    {
        m_nx = g.Nx()*g.nx(), m_ny = g.Ny()*g.ny();
        m_transpose = false;
        unsigned size1d = 0;
        if( direction == coo2d::x)
        {
            m_w=dg::construct<ContainerType>(dg::create::weights(g, {1,0}));
            dg::blas1::scal( m_temp, 1./g.lx());
            dg::blas1::scal( m_w, 1./g.lx());
            size1d = m_ny;
            if( "simple" == mode)
                dg::transpose( m_nx, m_ny, m_temp, m_w);
        }
        else
        {
            m_w=dg::construct<ContainerType>(dg::create::weights(g, {0,1}));
            m_transpose = true;
            dg::blas1::scal( m_temp, 1./g.ly());
            dg::blas1::scal( m_w, 1./g.ly());
            if( "exact" == mode)
                dg::transpose( m_nx, m_ny, m_temp, m_w);
            size1d = m_nx;
        }
        m_temp = m_w;
        thrust::host_vector<double> t1d( size1d);
        m_temp1d = dg::construct<ContainerType>( t1d);
        if( !("exact"==mode || "simple" == mode))
            throw dg::Error( dg::Message( _ping_) << "Mode must either be exact or simple!");

    }

    ///@copydoc Average()
    Average( const aTopology3d& g, enum coo3d direction, std::string mode = "exact"): m_mode(mode)
    {
        m_transpose = false;
        unsigned nx = g.nx()*g.Nx(), ny = g.ny()*g.Ny(), nz = g.nz()*g.Nz();
        if( direction == coo3d::x) {
            m_w = dg::construct<ContainerType>(dg::create::weights(g, {1,0,0}));
            dg::blas1::scal( m_temp, 1./g.lx());
            dg::blas1::scal( m_w, 1./g.lx());
            m_nx = nx, m_ny = ny*nz;
            if( "simple" == mode)
                dg::transpose( m_nx, m_ny, m_temp, m_w);
        }
        else if( direction == coo3d::z) {
            m_w = dg::construct<ContainerType>(dg::create::weights(g, {0,0,1}));
            m_transpose = true;
            dg::blas1::scal( m_temp, 1./g.lz());
            dg::blas1::scal( m_w, 1./g.lz());
            m_nx = nx*ny, m_ny = nz;
            if( "exact" == mode)
                dg::transpose( m_nx, m_ny, m_temp, m_w);
        }
        else if( direction == coo3d::xy) {
            m_w = dg::construct<ContainerType>(dg::create::weights(g, {1,1,0}));
            dg::blas1::scal( m_temp, 1./g.lx()/g.ly());
            dg::blas1::scal( m_w, 1./g.lx()/g.ly());
            m_nx = nx*ny, m_ny = nz;
            if( "simple" == mode)
                dg::transpose( m_nx, m_ny, m_temp, m_w);
        }
        else if( direction == coo3d::yz) {
            m_w = dg::construct<ContainerType>(dg::create::weights(g, {0,1,1}));
            m_transpose = true;
            dg::blas1::scal( m_temp, 1./g.ly()/g.lz());
            dg::blas1::scal( m_w, 1./g.ly()/g.lz());
            m_nx = nx, m_ny = ny*nz;
            if( "exact" == mode)
                dg::transpose( m_nx, m_ny, m_temp, m_w);
        }
        else
            std::cerr << "Warning: this direction is not implemented\n";
        m_temp = m_w;
        if(!m_transpose)
            m_temp1d = dg::construct<ContainerType>(
                thrust::host_vector<double>( m_ny,0.));
        else
            m_temp1d = dg::construct<ContainerType>(
                thrust::host_vector<double>( m_nx,0.));
        if( !("exact"==mode || "simple" == mode))
            throw dg::Error( dg::Message( _ping_) << "Mode must either be exact or simple!");
    }
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
        if( !m_transpose)
        {
            //temp1d has size m_ny
            if( "exact" == m_mode)
                dg::average( m_nx, m_ny, src, m_w, m_temp1d);
            else
            {
                dg::transpose( m_nx, m_ny, src, m_temp);
                dg::simple_average( m_ny, m_nx, m_temp, m_w, m_temp1d);
            }
            if( extend )
                dg::extend_column( m_nx, m_ny, m_temp1d, res);
            else
                res = m_temp1d;
        }
        else
        {
            //temp1d has size m_nx
            if( "exact" == m_mode)
            {
                dg::transpose( m_nx, m_ny, src, m_temp);
                dg::average( m_ny, m_nx, m_temp, m_w, m_temp1d);
            }
            else
                dg::simple_average( m_nx, m_ny, src, m_w, m_temp1d);
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
    std::string m_mode;
};


}//namespace dg
