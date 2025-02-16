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

///@addtogroup evaluation
///@{


/*! @brief Nodal weight coefficients
 *
 * %Equivalent to the following:
 *
 * -# from the given Nd dimensional grid generate Nd one-dimensional lists of
 *  integraion weights <tt> w_i = g.weights( i)</tt>
 * -# take the kronecker product of the 1d weights and store the result in the
 *  output vector <tt> w = dg::kronecker( dg::Product(), w_0, w_1, ...)</tt>
 *  The **0 dimension is the contiguous dimension** in the return vector \c w
 * .
 * For example
 * @snippet{trimleft} evaluation_t.cpp evaluate2d
 * @tparam Topology A fixed sized grid type with member functions <tt> static
 * constexpr size_t Topology::ndim()</tt> giving the number of dimensions and
 * <tt> vector_type Topology::weights( unsigned dim)</tt> giving the integration
 * weights in dimension \c dim
 * @param g The grid
 * @return The output vector \c w as a host vector. Its value type is the same
 * as the grid value type
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 */
template<class Topology>
auto weights( const Topology& g)
{
    return do_weights( g, std::make_index_sequence<Topology::ndim()>());
}

/*! @brief Nodal weight coefficients on a subset of dimensions
 *
 * %Equivalent to the following:
 *
 * -# from the given Nd dimensional grid generate Nd one-dimensional lists of
 *  integraion weights <tt> w_i = remains[i] ? g.weights( i) : 1</tt>
 * -# take the kronecker product of the 1d weights and store the result in the
 *  output vector <tt> w = dg::kronecker( dg::Product(), w_0, w_1, ...)</tt>
 *  The **0 dimension is the contiguous dimension** in the return vector \c w
 * .
 * @tparam Topology A fixed sized grid type with member functions <tt> static
 * constexpr size_t Topology::ndim()</tt> giving the number of dimensions and
 * <tt> vector_type Topology::weights( unsigned dim)</tt> giving the integration
 * weights in dimension \c dim
 * @param g The grid
 * @param remains For each dimension determine whether to use weights or a vector of 1s
 * @return Host Vector with full grid size
 * @note If you want the weights of the sub-grid that consists of the remaining dimensions
 * you need to manually create that sub-grid and use \c dg::create::weights( sub_grid)
 * The difference is the size of the resulting vector or the result of this function is
 * to prolongate the sub-grid weights to the full grid again
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 */
template<class Topology>
auto weights( const Topology& g, std::array<bool,Topology::ndim()> remains)
{
    return do_weights( g, remains, std::make_index_sequence<Topology::ndim()>());
}


/*! @brief Inverse nodal weight coefficients
 *
 * Short for
 * @code{.cpp}
 *   auto v = weights( g);
 *   dg::blas1::transform( v, v, dg::INVERT());
 *   return v;
 * @endcode
*/
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
