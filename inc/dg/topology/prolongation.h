#pragma once

#include "grid.h"
#include "weights.h"
#include "dg/blas1.h"
#include "interpolation.h"
#include "projection.h"
/*! @file

  @brief Prolongation matrix creation functions
  */

namespace dg{
namespace create{
///@cond
namespace detail{
template< class Topology, size_t M, size_t ...I>
auto do_full_abscissas( const Topology& g, std::array<unsigned, M> map,
        std::index_sequence<I...>)
{
    std::array< decltype(g.abscissas(0)),Topology::ndim()> abs;
    for( unsigned u=0; u<Topology::ndim(); u++)
        abs[u] = g.abscissas(u);
    auto abs_red = abs;
    for( unsigned u=0; u<Topology::ndim(); u++)
        dg::blas1::copy( 1, abs_red[u]);
    std::array<decltype(g.abscissas(0)),M> full_abs;
    for( unsigned u=0; u<M; u++)
    {
        dg::blas1::copy( abs[map[u]], abs_red[map[u]]);
        full_abs[u] = dg::kronecker( dg::Product(), abs_red[I]...);
        dg::blas1::copy( 1, abs_red[map[u]]);
    }
    return full_abs;
}

template<class Topology, size_t Md>
std::array<bool,Topology::ndim()> axes2bools( const Topology& g, std::array<unsigned,Md> axes)
{
    // The axes that are averaged
    std::array<bool,Topology::ndim()> bools;
    for( unsigned u=0; u<Topology::ndim(); u++)
        bools[u] = false;
    for( unsigned u=0; u<Md; u++)
    {
        assert( axes[u] < bools.size());
        bools[axes[u]] = true;
    }
    return bools;
}
template<class Topology, size_t Md>
std::array<unsigned,Topology::ndim()-Md> complement( const Topology& g, std::array<unsigned,Md> axes)
{
    // The axes that remain
    std::array<unsigned,Topology::ndim()-Md> comp = {0u};
    auto bools = axes2bools( g, axes);
    unsigned counter = 0;
    for ( unsigned u=0; u<Topology::ndim(); u++)
    {
        if( !bools[u])
        {
            comp[counter] = u;
            counter++;
        }
    }
    return comp;
}

} // namespace detail
///@endcond
///@addtogroup average
///@{

/*!@brief Prolongation matrix along given axes / Transpose of reduction
 *
 * For example if you have 2d vectors in the x-y plane that you
 * want to prolong along the z-direction:
 * @code{.cpp}
 * dg::Grid3d g3d( ...);
 * // Create prolongation along the 2 axis of g3d to g3d
 * auto prolong = dg::create::prolongation ( g3d, std::array{2u});
 * // Means you have vectors in the 0 and 1 axis
 * @endcode
 * @tparam Nd Prolongated grid number of dimenensions
 * @tparam Md Number of dimensions to prolongate \c Md<Nd
 * @param g_new the grid of the new, prolongated vectors
 * @param axes Axis numbers in \c g_new along which to prolong.
 * Does not need to be sorted but <tt>axes[i] < Nd</tt>
 * @return matrix that acts on vectors on the reduced grid <tt>g_new w/o
 * axes</tt> and produces vectors on \c g_new
 * @sa Prolongation is the transpose of a \c reduction and adjoint of \c projection
 */
template<class real_type, size_t Nd, size_t Md>
cusp::csr_matrix<int, real_type, cusp::host_memory> prolongation(
    const aRealTopology<real_type,Nd>& g_new,
    std::array<unsigned,Md> axes)
{
    static_assert( Md < Nd && Md > 0 && Nd > 0);
    std::array<unsigned,Nd-Md> remains = detail::complement( g_new, axes);
    auto full_abs = detail::do_full_abscissas( g_new, remains, std::make_index_sequence<Nd>());
    std::vector<cusp::csr_matrix<int,real_type,cusp::host_memory>> matrix(Nd-Md);
    for( unsigned u=0; u<Nd-Md; u++)
    {
        matrix[u] = detail::interpolation1d( dg::xspace, full_abs[u],
            g_new.grid(remains[u]), g_new.bc(remains[u]));
    }
    for( unsigned u=1; u<Nd-Md; u++)
        matrix[0] = dg::tensorproduct_cols( matrix[u], matrix[0]);
    return matrix[0];
}

/*!@brief Reduction matrix along given axes
 *
 * For example if you have 2d vectors in the x-y plane that you
 * want to reduce along the y-direction:
 * @code{.cpp}
 * dg::Grid2d g2d( ...);
 * auto reduce = dg::create::reduction ( std::array{1u}, g2d);
 * @endcode
 * @tparam Nd Full grid number of dimenensions
 * @tparam Md Number of dimensions to reduce \c Md<Nd
 * @param axes Axis numbers in \c g_old along which to reduce
 * @param g_old Grid of the old, un-reduced vectors
 * @return matrix that acts on vectors on the un-reduced grid \c g_old and
 * produces vectors on <tt>g_old w/o axes</tt>
 * @note Weights multiply through in 1/W R W
 * @sa projection Average
 * @sa Reduction is the transpose of a \c prolongation
 */
template<class real_type, size_t Nd, size_t Md>
cusp::coo_matrix<int, real_type, cusp::host_memory> reduction(
    std::array<unsigned,Md> axes,
    const aRealTopology<real_type,Nd>& g_old)
{
    cusp::coo_matrix<int, real_type, cusp::host_memory> temp = prolongation(
            g_old, axes), A;
    cusp::transpose( temp, A);
    A.sort_by_row_and_column();
    return A;
}

/*!@brief Projection matrix along given axes
 *
 * For example if you have 2d vectors in the x-y plane that you
 * want to reduce along the y-direction:
 * @code{.cpp}
 * dg::Grid2d g2d( ...);
 * auto project = dg::create::projection ( std::array{1u}, g2d);
 * @endcode
 * @tparam Nd Full grid number of dimenensions
 * @tparam Md Number of dimensions to reduce \c Md<Nd
 * @param axes Axis numbers in \c g_old along which to reduce
 * <tt>axes[i]<g_old.ndim()</tt>
 * @param g_old Grid of the old, un-reduced vectors
 * @return matrix that acts on vectors on the un-reduced grid \c g_old and
 * produces vectors on <tt>g_old w/o axes</tt>
 * @sa prolongation Average
 * @sa Projection is the adjoint of a \c prolongation
 */
template<class real_type, size_t Nd, size_t Md>
cusp::coo_matrix<int, real_type, cusp::host_memory> projection(
    std::array<unsigned,Md> axes,
    const aRealTopology<real_type,Nd>& g_old)
{
    std::array<unsigned,Nd-Md> remains = detail::complement( g_old, axes);
    std::array<RealGrid<real_type,1>,Nd-Md> gs;
    for( unsigned u=0; u<Nd-Md; u++)
        gs[u] = g_old.grid( remains[u]);
    RealGrid<real_type, Nd-Md> g_new(gs);
    auto w_old = dg::create::weights( g_old);
    auto v_new = dg::create::inv_weights( g_new);
    cusp::coo_matrix<int, real_type, cusp::host_memory> temp = prolongation(
            g_old, axes), A;
    cusp::transpose( temp, A);
    auto Wf = dg::create::diagonal( w_old);
    auto Vc = dg::create::diagonal( v_new);
    cusp::multiply( A, Wf, temp);
    cusp::multiply( Vc, temp, A);
    A.sort_by_row_and_column();
    return A;
}
///@}

} // namespace create

}//namespace dg
