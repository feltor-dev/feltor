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
auto do_full_abscissas( const Topology& g, std::array<unsigned, M> map, std::index_sequence<I...>)
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
        bools[axes[u]] = true;
    return bools;
}
template<class Topology, size_t Md>
std::array<unsigned,Topology::ndim()-Md> complement( const Topology& g, std::array<unsigned,Md> axes)
{
    // The axes that remain
    std::array<unsigned,Topology::ndim()-Md> comp;
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

// A prolongation in x means you have a vector in y
template<class real_type, size_t Nd, size_t Md>
cusp::csr_matrix<int, real_type, cusp::host_memory> prolongation(
    const aRealTopology<real_type,Nd>& g_new,
    std::array<unsigned,Md> axes)
{
    static_assert( Md <= Nd);
    std::array<unsigned,Nd-Md> remains = detail::complement( g_new, axes);
    auto full_abs = detail::do_full_abscissas( g_new, remains, std::make_index_sequence<Nd>());
    std::vector<cusp::csr_matrix<int,real_type,cusp::host_memory>> matrix(Nd-Md);
    for( unsigned u=0; u<Nd-Md; u++)
    {
        matrix[u] = detail::interpolation1d( full_abs[u], g_new.grid(remains[u]), g_new.bc(remains[u]));
    }
    for( unsigned u=1; u<Nd-Md; u++)
        matrix[0] = dg::tensorproduct_cols( matrix[u], matrix[0]);
    return matrix[0];
}

// TODO document that weights multiply through in 1/W R W
template<class real_type, size_t Nd, size_t Md>
cusp::coo_matrix<int, real_type, cusp::host_memory> reduction(
    std::array<unsigned,Md> axes,
    const aRealTopology<real_type,Nd>& g_old)
{
    static_assert( Md <= Nd);
    cusp::coo_matrix<int, real_type, cusp::host_memory> temp = prolongation( g_old, axes), A;
    cusp::transpose( temp, A);
    A.sort_by_row_and_column();
    return A;
}

template<class real_type, size_t Nd, size_t Md>
cusp::coo_matrix<int, real_type, cusp::host_memory> projection(
    std::array<unsigned,Md> axes,
    const aRealTopology<real_type,Nd>& g_old)
{
    std::array<unsigned,Nd-Md> remains = detail::complement( g_old, axes);
    std::array<RealGrid<real_type,1>,Nd-Md> gs;
    for( unsigned u=0; u<Nd-Md; u++)
        gs[u] = g_old.grid( remains[u]);
    RealGrid<real_type,Nd-Md> g_new(gs);
    auto w_old = dg::create::weights( g_old);
    auto v_new = dg::create::inv_weights( g_new);
    cusp::coo_matrix<int, real_type, cusp::host_memory> temp = prolongation( g_old, axes), A;
    cusp::transpose( temp, A);
    auto Wf = dg::create::diagonal( w_old);
    auto Vc = dg::create::diagonal( v_new);
    cusp::multiply( A, Wf, temp);
    cusp::multiply( Vc, temp, A);
    A.sort_by_row_and_column();
    return A;
}

} // namespace create

}//namespace dg
