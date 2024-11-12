#pragma once

#include <thrust/host_vector.h>
#include "grid.h"
#include "../functors.h"
#include "../blas1.h"
#include "../enums.h"

/*! @file

  * @brief Creation functions for integration weights and their inverse
  */

namespace dg{
namespace create{
///@cond
template< class Topology, size_t N, size_t ...I>
auto do_weights( const Topology& g, std::array<bool, N> coo, std::index_sequence<I...>)
{
    std::array< decltype(g.weights(0)),N> weights;
    for( unsigned u=0; u<N; u++)
        weights[u] = g.weights(u);
    for( unsigned u=0; u<N; u++)
        if( !coo[u])
            dg::blas1::copy( 1, weights[u]);
    return dg::kronecker( dg::Product(), weights[I]...);
}
template< class Topology, size_t ...I>
auto do_weights( const Topology& g, std::index_sequence<I...>)
{
    return dg::kronecker( dg::Product(), g.weights(I)...);
}

///@endcond

///@addtogroup highlevel
///@{

/*!@class hide_weights_doc
* @brief Nodal weight coefficients
* @param g The grid
* @return Host Vector
* @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
*/
/*!@class hide_inv_weights_doc
* @brief inverse nodal weight coefficients
* @param g The grid
* @return Host Vector
* @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
*/
/*!@class hide_weights_coo_doc
* @brief nodal weight coefficients
* @param g The grid
* @param coo The coordinate for which to generate the weights (in 2d only \c dg::x and \c dg::y are allowed)
* @return Host Vector with full grid size
* @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
*/

///@copydoc hide_weights_doc
///@copydoc hide_code_evaluate1d
///@copydoc hide_code_evaluate2d
template<class Topology>
auto weights( const Topology& g)
{
    return do_weights( g, std::make_index_sequence<Topology::ndim()>());
}

///@copydoc hide_weights_coo_doc
template<class Topology>
auto weights( const Topology& g, std::array<bool,Topology::ndim()> coo)
{
    return do_weights( g, coo, std::make_index_sequence<Topology::ndim()>());
}


///@copydoc hide_inv_weights_doc
template<class Topology>
auto inv_weights( const Topology& g)
{
    auto v = weights( g);
    dg::blas1::transform( v, v, dg::INVERT());
    return v;
}


///@}
}//namespace create
}//namespace dg
