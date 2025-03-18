#pragma once

#include <cusp/coo_matrix.h>
#include "xspacelib.h"
#ifdef MPI_VERSION
#include "mpi_projection.h" // for make_mpi_matrix function
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
    void set_boundary(
        cusp::array1d<real_type, cusp::host_memory>& values,
        cusp::array1d<int, cusp::host_memory>& column_indices,
        dg::bc bcx,
        int num_cols)
{
    for( unsigned k=0; k<values.size(); k++)
    {
        if( column_indices[k] < 0 )
        {
            if( bcx == dg::NEU || bcx == dg::NEU_DIR)
                column_indices[k] = -(column_indices[k]+1);
            else if( bcx == dg::DIR || bcx == dg::DIR_NEU)
            {
                column_indices[k] = -(column_indices[k]+1);
                values[k] *= -1;
            }
            else if( bcx == dg::PER)
                column_indices[k] += num_cols;
        }
        else if( column_indices[k] >= num_cols)
        {
            if( bcx == dg::NEU || bcx == dg::DIR_NEU)
                column_indices[k] = 2*num_cols-1-column_indices[k];
            else if( bcx == dg::DIR || bcx == dg::NEU_DIR)
            {
                column_indices[k] = 2*num_cols-1-column_indices[k];
                values[k] *= -1;
            }
            else if( bcx == dg::PER)
                column_indices[k] -= num_cols;
        }
    }
}

template<class real_type>
cusp::csr_matrix<int, real_type, cusp::host_memory> window_stencil(
        unsigned stencil_size,
        const RealGrid1d<real_type>& local,
        const RealGrid1d<real_type>& global,
        dg::bc bcx)
{
    cusp::array1d<real_type, cusp::host_memory> values;
    cusp::array1d<int, cusp::host_memory> row_offsets;
    cusp::array1d<int, cusp::host_memory> column_indices;

    unsigned num_rows = local.size();
    unsigned num_cols = global.size();
    unsigned radius = stencil_size/2;
    int L0 = round((local.x0() - global.x0())/global.h())*global.n();

    row_offsets.push_back(0);
    for( unsigned k=0; k<num_rows; k++)
    {
        row_offsets.push_back(stencil_size + row_offsets[k]);
        for( unsigned l=0; l<stencil_size; l++)
        {
            column_indices.push_back( L0 + (int)(k + l) - (int)radius);
            values.push_back( 1.0);
        }
    }
    set_boundary( values, column_indices, bcx, num_cols);

    cusp::csr_matrix<int, real_type, cusp::host_memory> A(
            num_rows, num_cols, values.size());
    A.row_offsets = row_offsets;
    A.column_indices = column_indices;
    A.values = values;
    return A;
}

// Rethink this approach if we ever need to make it work with MPI again
template<class real_type>
cusp::csr_matrix<int, real_type, cusp::host_memory> limiter_stencil(
        const RealGrid1d<real_type>& local,
        const RealGrid1d<real_type>& global,
        dg::bc bcx)
{
    cusp::array1d<real_type, cusp::host_memory> values;
    cusp::array1d<int, cusp::host_memory> row_offsets;
    cusp::array1d<int, cusp::host_memory> column_indices;

    unsigned num_rows = local.size();
    unsigned num_cols = global.size();
    int L0 = round((local.x0() - global.x0())/global.h())*global.n();
    // We need the first two lines of forward
    dg::SquareMatrix<real_type> forward = dg::DLT<real_type>::forward(local.n());
    // and the second column of backward
    dg::SquareMatrix<real_type> backward = dg::DLT<real_type>::backward(local.n());
    if( global.n() == 1)
        throw dg::Error( dg::Message(_ping_) << "Limiter stencil not possible for n==1!");


    row_offsets.push_back( 0);
    for( unsigned k=0; k<local.N(); k++)
    {
        row_offsets.push_back(row_offsets[row_offsets.size()-1] + 3*local.n());
        for( unsigned j=1; j<local.n(); j++)
            row_offsets.push_back(row_offsets[row_offsets.size()-1]);
        // Order is important
        for( unsigned j=0; j<local.n(); j++)
        {
            column_indices.push_back( L0 + (int)((k-1)*global.n() + j) );
            values.push_back( forward(0, j ));
        }
        for( unsigned j=0; j<local.n(); j++)
        {
            column_indices.push_back( L0 + (int)(k*global.n() + j ));
            values.push_back( forward(1, j ));
        }
        for( unsigned j=0; j<local.n(); j++)
        {
            column_indices.push_back( L0 + (int)((k+1)*global.n() + j));
            values.push_back( backward(j, 1) );
        }
    }
    assert( row_offsets.size() == num_rows+1);
    set_boundary( values, column_indices, bcx, num_cols);

    cusp::csr_matrix<int, real_type, cusp::host_memory> A(
            num_rows, num_cols, values.size());
    A.row_offsets = row_offsets;
    A.column_indices = column_indices;
    A.values = values;
    return A;
}

template<class real_type>
cusp::csr_matrix< int, real_type, cusp::host_memory> identity_matrix( const RealGrid1d<real_type>& local, const RealGrid1d<real_type>& global)
{
    cusp::csr_matrix<int,real_type,cusp::host_memory> A( local.size(),
            global.size(), local.size());
    int L0 = round((local.x0() - global.x0())/global.h())*global.n();
    A.row_offsets[0] = 0;
    for( unsigned i=0; i<local.size(); i++)
    {
        A.row_offsets[i+1] = 1 + A.row_offsets[i];
        A.column_indices[i] = L0 + i;
        A.values[i] = 1.;
    }
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
 * @sa \c dg::blas2::stencil
 */
template<class real_type>
dg::IHMatrix_t<real_type> window_stencil(
        unsigned window_size,
        const RealGrid1d<real_type>& g,
        dg::bc bcx)
{
    return detail::window_stencil( window_size, g, g, bcx);
}

/*!
 * @brief A stencil for the dg Slope limiter
 *
 * This stencil is specifically made to implement a dg slope limiter
 * @param g the grid, grid.n() must be greater than 0
 * @param bound Determine what to do at the boundary. For Neumann conditions the
 * boundary points are simply duplicated, For Dirichlet they are duplicated
 * as well and the values are multiplied with -1
 * @return A sparse matrix with 0 or 3n entries per row (the zero coefficient has 3n entries, the remaining coefficients have 0). The values contain the transformation matrices necessary to implement the filter
 * @tparam real_type The value type of the matrix
 * @sa \c dg::blas2::stencil \c dg::CSRSlopeLimiter
 * @attention Do not sort the matrix as then the ordering is destroyed. CANNOT BE USED IN MPI
 * @note Does unfortunately not work very well in more than 1 dimension
 */
template<class real_type>
dg::IHMatrix_t<real_type> limiter_stencil(
        const RealGrid1d<real_type>& g,
        dg::bc bound)
{
    return detail::limiter_stencil( g, g, bound);
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
 * @sa \c dg::blas2::stencil
 */
template<class real_type>
dg::IHMatrix_t<real_type> window_stencil(
        std::array<int,2> window_size,
        const aRealTopology2d<real_type>& g,
        dg::bc bcx, dg::bc bcy)
{
    auto mx = detail::window_stencil(window_size[0], g.gx(), g.gx(), bcx);
    auto my = detail::window_stencil(window_size[1], g.gy(), g.gy(), bcy);
    return dg::tensorproduct( my, mx);
}

///@copydoc limiter_stencil(const RealGrid1d<real_type>&,dg::bc)
///@param direction The limiter acts on only 1 direction at a time
template<class real_type>
dg::IHMatrix_t<real_type> limiter_stencil(
        enum coo3d direction,
        const aRealTopology2d<real_type>& g,
        dg::bc bound)
{
    if( direction == dg::coo3d::x)
    {
        auto mx = detail::limiter_stencil(g.gx(), g.gx(), bound);
        auto einsy = detail::identity_matrix( g.gy(), g.gy());
        return dg::tensorproduct( einsy, mx);
    }
    auto my = detail::limiter_stencil(g.gy(), g.gy(), bound);
    auto einsx = detail::identity_matrix( g.gx(), g.gx());
    return dg::tensorproduct( my, einsx);
}

///@copydoc limiter_stencil(const RealGrid1d<real_type>&,dg::bc)
///@param direction The limiter acts on only 1 direction at a time
template<class real_type>
dg::IHMatrix_t<real_type> limiter_stencil(
        enum coo3d direction,
        const aRealTopology3d<real_type>& g,
        dg::bc bound)
{
    if( direction == dg::coo3d::x)
    {
        auto mx = detail::limiter_stencil(g.gx(), g.gx(), bound);
        auto einsy = detail::identity_matrix( g.gy(), g.gy());
        auto einsz = detail::identity_matrix( g.gz(), g.gz());
        auto temp = dg::tensorproduct( einsy, mx);
        return dg::tensorproduct( einsz, temp);
    }
    if( direction == dg::coo3d::y)
    {
        auto einsx = detail::identity_matrix( g.gx(), g.gx());
        auto my = detail::limiter_stencil(g.gy(), g.gy(), bound);
        auto einsz = detail::identity_matrix( g.gz(), g.gz());
        return dg::tensorproduct( einsz, dg::tensorproduct( my, einsx));
    }
    auto mz = detail::limiter_stencil(g.gz(), g.gz(), bound);
    auto einsy = detail::identity_matrix( g.gy(), g.gy());
    auto einsx = detail::identity_matrix( g.gx(), g.gx());
    return dg::tensorproduct( mz, dg::tensorproduct( einsy, einsx));
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
 * @sa \c dg::blas2::stencil
 */
template<class real_type>
dg::IHMatrix_t<real_type> window_stencil(
        std::array<int,2> window_size,
        const aRealTopology3d<real_type>& g,
        dg::bc bcx, dg::bc bcy)
{
    auto mx = detail::window_stencil(window_size[0], g.gx(), g.gx(), bcx);
    auto my = detail::window_stencil(window_size[1], g.gy(), g.gy(), bcy);
    auto mz = detail::identity_matrix( g.gz(), g.gz());
    return dg::tensorproduct( mz, dg::tensorproduct( my, mx));
}

#ifdef MPI_VERSION
///@copydoc dg::create::window_stencil(std::array<int,2>,const aRealTopology2d<real_type>&,dg::bc,dg::bc)
template<class real_type>
dg::MIHMatrix_t<real_type> window_stencil(
        std::array<int,2> window_size,
        const aRealMPITopology2d<real_type>& g,
        dg::bc bcx, dg::bc bcy)
{
    auto mx = detail::window_stencil(window_size[0], g.local().gx(), g.global().gx(), bcx);
    auto my = detail::window_stencil(window_size[1], g.local().gy(), g.global().gy(), bcy);
    auto local = dg::tensorproduct( my, mx);
    return dg::make_mpi_matrix( local, g);
}

///@copydoc window_stencil(std::array<int,2>,const aRealTopology3d<real_type>&,dg::bc,dg::bc)
template<class real_type>
dg::MIHMatrix_t<real_type> window_stencil(
        std::array<int,2> window_size,
        const aRealMPITopology3d<real_type>& g,
        dg::bc bcx, dg::bc bcy)
{
    auto mx = detail::window_stencil(window_size[0], g.local().gx(), g.global().gx(), bcx);
    auto my = detail::window_stencil(window_size[1], g.local().gy(), g.global().gy(), bcy);
    auto mz = detail::identity_matrix( g.local().gz(), g.global().gz());
    auto out = dg::tensorproduct( mz, dg::tensorproduct( my, mx));
    return dg::make_mpi_matrix( out, g);
}

#endif // MPI_VERSION

///@}
} // namespace create
} // namespace dg
