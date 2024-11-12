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
template< class Topology, size_t N, size_t ...I>
auto do_full_abscissas( const Topology& g, std::vector<unsigned>& map, std::array<bool, N> coo, std::index_sequence<I...>)
{
    std::array< decltype(g.abscissas(0)),N> abs;
    for( unsigned u=0; u<N; u++)
        abs[u] = g.abscissas(u);
    auto abs_red = abs;
    for( unsigned u=0; u<N; u++)
        dg::blas1::copy( 1, abs_red[u]);
    std::vector<decltype(g.abscissas(0))> full_abs;
    for( unsigned u=0; u<N; u++)
        if ( coo[u] )
        {
            map.push_back(u);
            dg::blas1::copy( abs[u], abs_red[u]);
            full_abs.push_back( dg::kronecker( dg::Product(), abs_red[I]...));
            dg::blas1::copy( 1, abs_red[u]);
        }

    return full_abs;
}
} // namespace detail
///@endcond

template<class real_type, size_t Nd>
cusp::csr_matrix<int, real_type, cusp::host_memory> prolongation(
    const aRealTopology<real_type,Nd>& g_new,
    std::array<bool,Nd> remains)
{
    std::vector<unsigned> map; // maps index of full_abs to abscissa in grid
    auto full_abs = detail::do_full_abscissas( g_new, map, remains, std::make_index_sequence<Nd>());
    size_t Md = full_abs.size();
    std::vector<cusp::csr_matrix<int,real_type,cusp::host_memory>> axes(Md);
    for( unsigned u=0; u<Md; u++)
    {
        axes[u] = detail::interpolation1d( full_abs[u], g_new.grid(map[u]), g_new.bc(u));
    }
    for( unsigned u=1; u<Md; u++)
        axes[0] = dg::tensorproduct_cols( axes[u], axes[0]);
    return axes[0];
}
template<class real_type, size_t Nd>
cusp::coo_matrix<int, real_type, cusp::host_memory> average(
    std::array<bool,Nd> remains,
    const aRealTopology<real_type,Nd>& g_old)
{
    //form the adjoint
    cusp::coo_matrix<int, real_type, cusp::host_memory> Wf =
        dg::create::diagonal( dg::create::weights( g_old));
    std::vector<thrust::host_vector<real_type>> w_remains;
    for( unsigned u=0; u<Nd; u++)
        if( remains[u])
            w_remains.push_back( g_old.grid(u).weights());

    size_t Md = w_remains.size();
    for( unsigned u=1; u<Md; u++)
        w_remains[0] = dg::kronecker( dg::Product(), w_remains[0], w_remains[u]);

    dg::blas1::transform( w_remains[0], w_remains[0], dg::INVERT());
    cusp::coo_matrix<int, real_type, cusp::host_memory> Vc =
        dg::create::diagonal( w_remains[0]);
    cusp::coo_matrix<int, real_type, cusp::host_memory> temp = prolongation( g_old, remains), A;
    cusp::transpose( temp, A);
    //!!! cusp::multiply removes explicit zeros in the output
    cusp::multiply( A, Wf, temp);
    cusp::multiply( Vc, temp, A);
    A.sort_by_row_and_column();
    return A;

}//namespace create
} // namespace create

}//namespace dg
