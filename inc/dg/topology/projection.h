#pragma once
#include <vector>
#include "grid.h"
#include "interpolation.h"
#include "weights.h"
#include "fem.h"

/*!@file

  @brief Creation of projection matrices
 */
namespace dg{

/**
 * @brief Greatest common divisor
 *
 * @param a First number
 * @param b Second number
 *
 * @return greatest common divisor
 * @ingroup basics
 */
template<class T>
T gcd( T a, T b)
{
    T r2 = std::max(a,b);
    T r1 = std::min(a,b);
    while( r1!=0)
    {
        r2 = r2%r1;
        std::swap( r1, r2);
    }
    return r2;
}

/**
 * @brief Least common multiple
 *
 * @param a Fist number
 * @param b Second number
 *
 * @return Least common multiple
 * @ingroup basics
 */
template<class T>
T lcm( T a, T b)
{
    T g = gcd( a,b);
    return a/g*b;
}

namespace create{
///@addtogroup interpolation
///@{

/**
 * @brief Create a diagonal matrix
 *
 * This matrix is given by \f$ D_{ij} = d_i \delta_{ij}\f$
 * @param diagonal The diagonal elements d_i
 * @return diagonal matrix
 */
template<class real_type>
dg::SparseMatrix< int, real_type, thrust::host_vector> diagonal( const thrust::host_vector<real_type>& diagonal)
{
    unsigned size = diagonal.size();
    thrust::host_vector<int> A_row_offsets(size+1), A_column_indices( size);
    thrust::host_vector<real_type> A_values( size);

    A_row_offsets[0] = 0;
    for( unsigned i=0; i<size; i++)
    {
        A_row_offsets[i+1] = i+1;
        A_column_indices[i] = i;
        A_values[i] = diagonal[i];
    }
    return { size, size, A_row_offsets, A_column_indices, A_values};
}


/**
 * @brief Create a projection between two grids
 *
 * This matrix can be applied to vectors defined on the old (fine) grid to
 * obtain its values projected on the new (coarse) grid. (Projection means that
 * the projection integrals over the base polynomials are computed).  If the
 * fine grid is a multiple of the coarse grid, the integral value of the
 * projected vector will be conserved and the difference in the L2 norm between
 * old and new vector small.  The projection matrix is the adjoint of the
 * interpolation matrix
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 * @sa for integer multiples between old and new grid you may want to consider the dg::create::fast_projection functions
 *
 * @param g_new The new (coarse) grid
 * @param g_old The old (fine) grid
 * @copydoc hide_method
 *
 * @return Projection matrix
 * @note The boundaries of the old grid must lie within the boundaries of the
 * new grid
 * @note also check \c dg::create::transformation, which is the more general
 * solution
 * @attention Projection only works if the number of cells in the fine grid is
 * a multiple of the number of cells in the coarse grid and if the number of
 * polynomial coefficients is lower or the same in the new grid
 */
template<class real_type, size_t Nd>
dg::SparseMatrix< int, real_type, thrust::host_vector> projection(
    const aRealTopology<real_type,Nd>& g_new,
    const aRealTopology<real_type,Nd>& g_old, std::string method = "dg")
{
    // These tests cannot pass for MPI
    //for( unsigned u=0; u<Nd; u++)
    //{
    //    if( fabs(g_new.h(u) / g_old.h(u) -round(g_new.h(u) / g_old.h(u))) > 1e-13)
    //        throw dg::Error( dg::Message(_ping_)
    //                  << "# ATTENTION: you project between incompatible grids!! old h: "
    //                  <<g_old.h(u)<<" new h: "<<g_new.h(u)<<"\n");
    //    // Check that boundaries of old grid conincide with cell boundaries of new
    //    real_type fp = (g_old.p(u) - g_new.p(u) ) / g_new.h(u); // must be integer
    //    if( fabs(fp - round( fp)) > 1e-13 and method == "dg")
    //        throw dg::Error( dg::Message(_ping_)
    //                  << "# ATTENTION: you project between incompatible grids!! old p: "
    //                  <<g_old.p(u)<<" new p: "<<g_new.p(u)<<" fp "<<fp<<"\n");
    //    real_type fq = (g_old.q(u) - g_new.p(u) ) / g_new.h(u); // must be integer
    //    if( fabs(fq - round( fq)) > 1e-13 and method == "dg")
    //        throw dg::Error( dg::Message(_ping_)
    //                  << "# ATTENTION: you project between incompatible grids!! old q: "
    //                  <<g_old.q(u)<<" new p: "<<g_new.p(u)<<" fq "<<fq<<"\n");
    //    if( g_old.n(u) < g_new.n(u))
    //        throw dg::Error( dg::Message(_ping_)
    //                   << "# ATTENTION: you project between incompatible grids!! old n: "
    //                   <<g_old.n(u)<<" new n: "<<g_new.n(u)<<"\n");
    //}
    //form the adjoint
    auto w_old = dg::create::weights( g_old);
    auto v_new = dg::create::inv_weights( g_new);
    auto project = interpolation( g_old, g_new, method).transpose();
    for( unsigned row=0; row<project.num_rows(); row++)
        for ( int jj = project.row_offsets()[row]; jj < project.row_offsets()[row+1]; jj++)
        {
            int col = project.column_indices()[jj];
            // Note that w_old is multiplied before v_new (keeps results backwards reproducible)
            project.values()[jj] = v_new[row] * ( project.values()[jj]* w_old[col]);
        }
    return project;
}

/**
 * @brief Create a transformation matrix between two grids
 *
 * The transformation matrix is probably the most correct way of
 transforming dG vectors between any two grids of different resolution.
 It first finds the least common multiple grid (lcm) of the old and the new grid. Then
 it interpolates the values to the lcm grid and finally projects them back to
 the new grid. In total we have
 \f[
 \mathcal T = P Q
 \f]
 where \f$ Q\f$ is the interpolation matrix and \f$ P \f$ the projection. If either new or
 old grid is already the lcm grid this function reduces to the interpolation/projection function.
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 *
 * @param g_new The new grid
 * @param g_old The old grid
 *
 * @return transformation matrix
 * @note The boundaries of the old grid must lie within the boundaries of the new grid
 * @note If the grid are very incompatible the matrix-matrix multiplication can take a while
 */
template<class real_type, size_t Nd>
dg::SparseMatrix< int, real_type, thrust::host_vector> transformation(
    const aRealTopology<real_type,Nd>& g_new,
    const aRealTopology<real_type,Nd>& g_old)
{
    std::array<unsigned, Nd> n_lcm, N_lcm;
    for( unsigned u=0; u<Nd; u++)
    {
        n_lcm [u] = lcm( g_new.n(u), g_old.n(u));
        N_lcm [u] = lcm( g_new.N(u), g_old.N(u));
    }
    RealGrid<real_type, Nd> g_lcm ( g_new.get_p(), g_new.get_q(), n_lcm, N_lcm, g_new.get_bc());
    return create::projection( g_new, g_lcm)*create::interpolation( g_lcm, g_old);
}

///@}
///@addtogroup scatter
///@{

/**
 * @brief Create a matrix \f$ PI\f$ that projects values to an equidistant grid
 *
 * Same as <tt>dg::create::transformation( g_equidist, g)</tt>
 * @param g The grid on which to operate
 *
 * @return transformation matrix (block diagonal)
 * @sa dg::create::backscatter, dg::create::transformation
 */
template<class real_type, size_t Nd>
dg::IHMatrix_t<real_type> backproject( const aRealTopology<real_type,Nd>& g)
{
    std::array<dg::IHMatrix_t<real_type>,Nd> matrix;
    for( unsigned u=0; u<Nd; u++)
    {
        unsigned n=g.n(u);
        dg::RealGrid1d<real_type> g_old( -1., 1., n, 1);
        dg::RealGrid1d<real_type> g_new( -1., 1., 1, n);
        auto block = dg::create::transformation( g_new, g_old);
        dg::SquareMatrix<real_type> op(n, 0.);
        for( unsigned i=0; i<block.num_rows(); i++)
            for( unsigned j=block.row_offsets()[i]; j<(unsigned)block.row_offsets()[i+1]; j++)
                op( i, block.column_indices()[j]) = block.values()[j];
        matrix[u] = (dg::IHMatrix_t<real_type>)dg::tensorproduct( g.N(u), op);

    }
    for( unsigned u=1; u<Nd; u++)
        matrix[0] = dg::tensorproduct( matrix[u], matrix[0]);
    return matrix[0];
}

/**
 * @brief Create a matrix \f$ (PI)^{-1}\f$ that transforms values from an equidistant grid back to a dg grid
 *
 * Same as <tt>dg::create::transformation( g, g_equidist)</tt>
 * @note The inverse of the backproject matrix is **not** its adjoint!
 * @param g The grid on which to operate
 *
 * @return transformation matrix (block diagonal)
 * @sa dg::create::inv_backscatter dg::create::backproject
 */
template<class real_type, size_t Nd>
dg::IHMatrix_t<real_type> inv_backproject( const aRealTopology<real_type,Nd>& g)
{
    std::array<dg::IHMatrix_t<real_type>,Nd> matrix;
    for( unsigned u=0; u<Nd; u++)
    {
        unsigned n=g.n(u);
        dg::RealGrid1d<real_type> g_old( -1., 1., n, 1);
        dg::RealGrid1d<real_type> g_new( -1., 1., 1, n);
        auto block = dg::create::transformation( g_new, g_old);
        dg::SquareMatrix<real_type> op(n, 0.);
        for( unsigned i=0; i<block.num_rows(); i++)
            for( unsigned j=block.row_offsets()[i]; j<(unsigned)block.row_offsets()[i+1]; j++)
                op( i, block.column_indices()[j]) = block.values()[j];
        matrix[u] = (dg::IHMatrix_t<real_type>)dg::tensorproduct( g.N(u), dg::invert(op));

    }
    for( unsigned u=1; u<Nd; u++)
        matrix[0] = dg::tensorproduct( matrix[u], matrix[0]);
    return matrix[0];
}

///@}

}//namespace create
}//namespace dg
