#pragma once

#include <cusp/coo_matrix.h>
#include <cusp/print.h>
#include "xspacelib.h"
#ifdef MPI_VERSION
#include "mpi_projection.h" // for convert function
#endif // MPI_VERSION

/*! @file
  @brief Stencil generation
  */
namespace dg
{
namespace create
{
///@cond
namespace detail
{
template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> stencil(
        unsigned stencil_size,
        const RealGrid1d<real_type>& stencil_area,
        const RealGrid1d<real_type>& g,
        dg::bc bcx = dg::NEU)
{
    cusp::array1d<real_type, cusp::host_memory> values;
    cusp::array1d<int, cusp::host_memory> row_indices;
    cusp::array1d<int, cusp::host_memory> column_indices;

    unsigned num_rows = stencil_area.size();
    unsigned num_cols = g.size();
    unsigned radius = stencil_size/2;
    int L0 = round((stencil_area.x0() - g.x0())/g.h())*g.n();

    for( unsigned k=0; k<num_rows; k++)
    {
        for( unsigned l=0; l<stencil_size; l++)
        {
            row_indices.push_back( k);
            column_indices.push_back( L0 + (int)(k + l) - (int)radius);
            values.push_back( 1.0);
        }
    }
    for( unsigned k=0; k<values.size(); k++)
    {
        if( column_indices[k] < 0 )
        {
            if( bcx == dg::NEU || bcx == dg::NEU_DIR)
                column_indices[k] = -(column_indices[k]+1);
            else if( bcx == dg::DIR || bcx == dg::DIR_NEU)
            {
                column_indices[k] = -(column_indices[k]+1);
                values[k] = -1;
            }
            else if( bcx == dg::PER)
                column_indices[k] += num_cols;
        }
        else if( column_indices[k] >= (int)num_cols)
        {
            if( bcx == dg::NEU || bcx == dg::DIR_NEU)
                column_indices[k] = 2*num_cols-1-column_indices[k];
            else if( bcx == dg::DIR || bcx == dg::NEU_DIR)
            {
                column_indices[k] = 2*num_cols-1-column_indices[k];
                values[k] = -1;
            }
            else if( bcx == dg::PER)
                column_indices[k] -= num_cols;
        }
    }

    cusp::coo_matrix<int, real_type, cusp::host_memory> A(
            num_rows, num_cols, values.size());
    A.row_indices = row_indices;
    A.column_indices = column_indices;
    A.values = values;
    return A;
}
} //namespace detail
///@endcond

///@addtogroup stencil
///@{

/*!
 * @brief A 1d centered window stencil
 *
 * Create a CSR Matrix containing a centered fixed sized window on each row.
 * @param window_size The number of points involved in the window. If even, the
 * number of points left is 1 higher than right.
 * @param g the grid
 * @param bcx Determine what to do at the boundary. For Neumann conditions the
 * boundary points are simply duplicated, For Dirichlet they are duplicated
 * as well and the values are set to -1 instead of 1.
 * @return A sparse matrix with \c window_size entries per row, each with value 1
 * @tparam real_type The value type of the matrix
 * @sa \c dg::blas2::filtered_symv
 */
template<class real_type>
dg::IHMatrix_t<real_type> window_stencil(
        unsigned window_size,
        const RealGrid1d<real_type>& g,
        dg::bc bcx = dg::NEU)
{
    return detail::stencil( window_size, g, g, bcx);
}

/*!
 * @brief A 2d centered window stencil
 *
 * Create a CSR Matrix containing a centered fixed sized window on each row
 * as the tensor product of two 1d stencils.
 * @param window_size The number of points involved in the window in each dimension.
 * First entry is x-dimension, 2nd is y-dimension.
 * If even, the number of points left is 1 higher than right.
 * @param g the grid
 * @param bcx Determine what to do at the x-boundary. For Neumann conditions the
 * boundary points are simply duplicated, For Dirichlet they are duplicated
 * as well and the values are set to -1 instead of 1.
 * @param bcy Determine what to do at the y-boundary. For Neumann conditions the
 * boundary points are simply duplicated, For Dirichlet they are duplicated
 * as well and the values are set to -1 instead of 1.
 * @return A sparse matrix with <tt> window_size[0]*window_size[1] </tt> entries per row, each with value 1
 * @tparam real_type The value type of the matrix
 * @sa \c dg::blas2::filtered_symv
 */
template<class real_type>
dg::IHMatrix_t<real_type> window_stencil(
        std::array<int,2> window_size,
        const aRealTopology2d<real_type>& g,
        dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU)
{
    auto mx = detail::stencil(window_size[0], g.gx(), g.gx(), bcx);
    auto my = detail::stencil(window_size[1], g.gy(), g.gy(), bcy);
    return dg::tensorproduct( my, mx);
}

/*!
 * @brief A 2d centered window stencil
 *
 * Create a CSR Matrix containing a centered fixed sized window on each row
 * as the tensor product of two 1d stencils and the identity in the third dimension
 * @param window_size The number of points involved in the window in each dimension.
 * First entry is x-dimension, 2nd is y-dimension.
 * If even, the number of points left is 1 higher than right.
 * @param g the grid
 * @param bcx Determine what to do at the x-boundary. For Neumann conditions the
 * boundary points are simply duplicated, For Dirichlet they are duplicated
 * as well and the values are set to -1 instead of 1.
 * @param bcy Determine what to do at the y-boundary. For Neumann conditions the
 * boundary points are simply duplicated, For Dirichlet they are duplicated
 * as well and the values are set to -1 instead of 1.
 * @return A sparse matrix with <tt> window_size[0]*window_size[1] </tt> entries per row, each with value 1
 * @tparam real_type The value type of the matrix
 * @sa \c dg::blas2::filtered_symv
 */
template<class real_type>
dg::IHMatrix_t<real_type> window_stencil(
        std::array<int,2> window_size,
        const aRealTopology3d<real_type>& g,
        dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU)
{
    auto mx = detail::stencil(window_size[0], g.gx(), g.gx(), bcx);
    auto my = detail::stencil(window_size[1], g.gy(), g.gy(), bcy);
    unsigned Nz = g.gz().size();
    cusp::coo_matrix<int,real_type,cusp::host_memory> mz( Nz, Nz, Nz);
    for( unsigned i=0; i<Nz; i++)
    {
        mz.row_indices[i] = mz.column_indices[i] = i;
        mz.values[i] = 1.;
    }

    auto two =  dg::tensorproduct( my, mx);
    return dg::tensorproduct( mz, two);
}

#ifdef MPI_VERSION
///@copydoc dg::create::window_stencil(std::array<int,2>,const aRealTopology2d<real_type>&,dg::bc,dg::bc)
template<class real_type>
dg::MIHMatrix_t<real_type> window_stencil(
        std::array<int,2> window_size,
        const aRealMPITopology2d<real_type>& g,
        dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU)
{
    auto mx = detail::stencil(window_size[0], g.local().gx(), g.global().gx(), bcx);
    auto my = detail::stencil(window_size[1], g.local().gy(), g.global().gy(), bcy);
    auto local = dg::tensorproduct( my, mx);
    return dg::convert( (dg::IHMatrix)local, g);
}

///@copydoc window_stencil(std::array<int,2>,const aRealTopology3d<real_type>&,dg::bc,dg::bc)
template<class real_type>
dg::MIHMatrix_t<real_type> window_stencil(
        std::array<int,2> window_size,
        const aRealMPITopology3d<real_type>& g,
        dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU)
{
    auto mx = detail::stencil(window_size[0], g.local().gx(), g.global().gx(), bcx);
    auto my = detail::stencil(window_size[1], g.local().gy(), g.global().gy(), bcy);
    auto local = dg::tensorproduct( my, mx);
    unsigned localNz = g.local().Nz()*g.nz();
    unsigned globalNz = g.global().Nz()*g.nz();
    cusp::coo_matrix<int,real_type,cusp::host_memory> mz( localNz, globalNz, localNz);
    int L0 = round((g.local().z0() - g.z0())/g.hz())*g.nz();
    for( unsigned i=0; i<localNz; i++)
    {
        mz.row_indices[i] = i;
        mz.column_indices[i] = L0 + i;
        mz.values[i] = 1.;
    }
    auto two =  dg::tensorproduct( my, mx);
    auto three = dg::tensorproduct( mz, two);
    return dg::convert( (dg::IHMatrix)three, g);
}

#endif // MPI_VERSION

///@}
} // namespace create
} // namespace dg
