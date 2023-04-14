#pragma once
#include <vector>
#include <cusp/coo_matrix.h>
#include <cusp/transpose.h>
#include "grid.h"
#include "interpolation.h"
#include "weights.h"
#include "fem.h"

/*!@file

  @brief Creation of projection matrices
 */
namespace dg{
///@addtogroup interpolation
///@{

/**
 * @brief Greatest common divisor
 *
 * @param a First number
 * @param b Second number
 *
 * @return greatest common divisor
 * @ingroup misc
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
 * @ingroup misc
 */
template<class T>
T lcm( T a, T b)
{
    T g = gcd( a,b);
    return a/g*b;
}

namespace create{

/**
 * @brief Create a diagonal matrix
 *
 * This matrix is given by \f$ D_{ij} = d_i \delta_{ij}\f$
 * @param diagonal The diagonal elements d_i
 * @return diagonal matrix
 */
template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> diagonal( const thrust::host_vector<real_type>& diagonal)
{
    unsigned size = diagonal.size();
    cusp::coo_matrix<int, real_type, cusp::host_memory> W( size, size, size);
    for( unsigned i=0; i<size; i++)
    {
        W.row_indices[i] = W.column_indices[i] = i;
        W.values[i] = diagonal[i];
    }
    return W;
}


/**
 * @brief Create a projection between two grids
 *
 * This matrix can be applied to vectors defined on the old (fine) grid to obtain
 * its values projected on the new (coarse) grid. (Projection means that the
 * projection integrals over the base polynomials are computed).
 * If the fine grid is a multiple of the coarse grid, the integral value
 of the projected vector will be conserved and the difference in the L2 norm
 between old and new vector small.
 * The projection matrix is the adjoint of the interpolation matrix
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 * @sa for integer multiples between old and new grid you may want to consider the dg::create::fast_projection functions
 *
 * @param g_new The new (coarse) grid
 * @param g_old The old (fine) grid
 * @copydoc hide_method
 *
 * @return Projection matrix
 * @note The boundaries of the old grid must lie within the boundaries of the new grid
 * @note also check \c dg::create::transformation, which is the more general solution
 * @attention Projection only works if the number of cells in the
 * fine grid is a multiple of the number of cells in the coarse grid
 * and if the number of polynomial coefficients is lower or the same in the new grid
 */
template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> projection( const RealGrid1d<real_type>& g_new, const RealGrid1d<real_type>& g_old, std::string method = "dg")
{
    if( g_old.N() % g_new.N() != 0) std::cerr << "# ATTENTION: you project between incompatible grids!! old N: "<<g_old.N()<<" new N: "<<g_new.N()<<"\n";
    if( g_old.n() < g_new.n()) std::cerr << "# ATTENTION: you project between incompatible grids!! old n: "<<g_old.n()<<" new n: "<<g_new.n()<<"\n";
    //form the adjoint
    cusp::coo_matrix<int, real_type, cusp::host_memory> Wf =
        dg::create::diagonal( dg::create::weights( g_old));
    cusp::coo_matrix<int, real_type, cusp::host_memory> Vc =
        dg::create::diagonal( dg::create::inv_weights( g_new));
    cusp::coo_matrix<int, real_type, cusp::host_memory> temp = interpolation( g_old, g_new, method), A;
    cusp::transpose( temp, A);
    //!!! cusp::multiply removes explicit zeros in the output
    cusp::multiply( A, Wf, temp);
    cusp::multiply( Vc, temp, A);
    A.sort_by_row_and_column();
    return A;
}


///@copydoc projection(const RealGrid1d&,const RealGrid1d&,std::string)
template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> projection( const aRealTopology2d<real_type>& g_new, const aRealTopology2d<real_type>& g_old, std::string method = "dg")
{
    cusp::csr_matrix<int, real_type, cusp::host_memory> projectX = projection( g_new.gx(), g_old.gx(), method);
    cusp::csr_matrix<int, real_type, cusp::host_memory> projectY = projection( g_new.gy(), g_old.gy(), method);
    return dg::tensorproduct( projectY, projectX);
}

///@copydoc projection(const RealGrid1d&,const RealGrid1d&,std::string)
template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> projection( const aRealTopology3d<real_type>& g_new, const aRealTopology3d<real_type>& g_old, std::string method = "dg")
{
    cusp::csr_matrix<int, real_type, cusp::host_memory> projectX = projection( g_new.gx(), g_old.gx(), method);
    cusp::csr_matrix<int, real_type, cusp::host_memory> projectY = projection( g_new.gy(), g_old.gy(), method);
    cusp::csr_matrix<int, real_type, cusp::host_memory> projectZ = projection( g_new.gz(), g_old.gz(), method);
    return dg::tensorproduct( projectZ, dg::tensorproduct( projectY, projectX));
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
template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> transformation( const aRealTopology3d<real_type>& g_new, const aRealTopology3d<real_type>& g_old)
{
    Grid3d g_lcm( Grid1d( g_new.x0(), g_new.x1(), lcm( g_new.nx(), g_old.nx()), lcm( g_new.Nx(), g_old.Nx())),
                  Grid1d( g_new.y0(), g_new.y1(), lcm( g_new.ny(), g_old.ny()), lcm( g_new.Ny(), g_old.Ny())),
                  Grid1d( g_new.z0(), g_new.z1(), lcm( g_new.nz(), g_old.nz()), lcm( g_new.Nz(), g_old.Nz())));
    cusp::coo_matrix< int, real_type, cusp::host_memory> Q = create::interpolation( g_lcm, g_old);
    cusp::coo_matrix< int, real_type, cusp::host_memory> P = create::projection( g_new, g_lcm), Y;
    cusp::multiply( P, Q, Y);
    Y.sort_by_row_and_column();
    return Y;
}

///@copydoc transformation(const aRealTopology3d&,const aRealTopology3d&)
template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> transformation( const aRealTopology2d<real_type>& g_new, const aRealTopology2d<real_type>& g_old)
{
    Grid2d g_lcm( Grid1d( g_new.x0(), g_new.x1(), lcm( g_new.nx(), g_old.nx()), lcm( g_new.Nx(), g_old.Nx())),
                  Grid1d( g_new.y0(), g_new.y1(), lcm( g_new.ny(), g_old.ny()), lcm( g_new.Ny(), g_old.Ny())));
    cusp::coo_matrix< int, real_type, cusp::host_memory> Q = create::interpolation( g_lcm, g_old);
    cusp::coo_matrix< int, real_type, cusp::host_memory> P = create::projection( g_new, g_lcm), Y;
    cusp::multiply( P, Q, Y);
    Y.sort_by_row_and_column();
    return Y;
}
///@copydoc transformation(const aRealTopology3d&,const aRealTopology3d&)
template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> transformation( const RealGrid1d<real_type>& g_new, const RealGrid1d<real_type>& g_old)
{
    RealGrid1d<real_type> g_lcm(g_new.x0(), g_new.x1(), lcm(g_new.n(), g_old.n()), lcm(g_new.N(), g_old.N()));
    cusp::coo_matrix< int, real_type, cusp::host_memory> Q = create::interpolation( g_lcm, g_old);
    cusp::coo_matrix< int, real_type, cusp::host_memory> P = create::projection( g_new, g_lcm), Y;
    cusp::multiply( P, Q, Y);
    Y.sort_by_row_and_column();
    return Y;
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
template<class real_type>
dg::IHMatrix_t<real_type> backproject( const RealGrid1d<real_type>& g)
{
    unsigned n=g.n();
    dg::RealGrid1d<real_type> g_old( -1., 1., n, 1);
    dg::RealGrid1d<real_type> g_new( -1., 1., 1, n);
    auto block = dg::create::transformation( g_new, g_old);
    dg::Operator<real_type> op(n, 0.);
    for( unsigned i=0; i<block.num_entries; i++)
        op( block.row_indices[i], block.column_indices[i]) = block.values[i];

    return (dg::IHMatrix_t<real_type>)dg::tensorproduct( g.N(), op);
}

///@copydoc backproject(const RealGrid1d<real_type>&)
template<class real_type>
dg::IHMatrix_t<real_type> backproject( const aRealTopology2d<real_type>& g)
{
    auto transformX = backproject( g.gx());
    auto transformY = backproject( g.gy());
    return dg::tensorproduct( transformY, transformX);
}

///@copydoc backproject(const RealGrid1d<real_type>&)
template<class real_type>
dg::IHMatrix_t<real_type> backproject( const aRealTopology3d<real_type>& g)
{
    auto transformX = backproject( g.gx());
    auto transformY = backproject( g.gy());
    auto transformZ = backproject( g.gz());
    return dg::tensorproduct( transformZ, dg::tensorproduct(transformY, transformX));
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
template<class real_type>
dg::IHMatrix_t<real_type> inv_backproject( const RealGrid1d<real_type>& g)
{
    unsigned n=g.n();
    dg::RealGrid1d<real_type> g_old( -1., 1., n, 1);
    dg::RealGrid1d<real_type> g_new( -1., 1., 1, n);
    auto block = dg::create::transformation( g_new, g_old);
    dg::Operator<real_type> op(n, 0.);
    for( unsigned i=0; i<block.num_entries; i++)
        op( block.row_indices[i], block.column_indices[i]) = block.values[i];

    return (dg::IHMatrix_t<real_type>)dg::tensorproduct( g.N(), dg::invert(op));
}

///@copydoc inv_backproject(const RealGrid1d<real_type>&)
template<class real_type>
dg::IHMatrix_t<real_type> inv_backproject( const aRealTopology2d<real_type>& g)
{
    //create equidistant backward transformation
    auto transformX = inv_backproject( g.gx());
    auto transformY = inv_backproject( g.gy());
    return dg::tensorproduct( transformY, transformX);
}

///@copydoc inv_backproject(const RealGrid1d<real_type>&)
template<class real_type>
dg::IHMatrix_t<real_type> inv_backproject( const aRealTopology3d<real_type>& g)
{
    auto transformX = inv_backproject( g.gx());
    auto transformY = inv_backproject( g.gy());
    auto transformZ = inv_backproject( g.gz());
    return dg::tensorproduct( transformZ, dg::tensorproduct(transformY, transformX));
}

///@}

}//namespace create
}//namespace dg
