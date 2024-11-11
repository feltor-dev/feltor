#pragma once

#include <cassert>
#include <cmath>
#include <thrust/host_vector.h>
#include "dg/backend/config.h"
#include "grid.h"
#include "operator.h"

/*! @file
  @brief Function discretization routines
  */
namespace dg
{

///@cond
template< class Functor, class Topology, size_t ...I>
auto do_evaluate( Functor f, const Topology& g, std::index_sequence<I...>)
{
    auto abs = g.abscissas();
    return dg::kronecker( f, abs[I]...);
}

///@endcond
///@addtogroup evaluation
///@{

/**
 * @brief %Evaluate a function on grid coordinates
 *
 * %Evaluate is equivalent to the following:
 *
 * -# generate a list of grid coordinates \f$ x_i, ...\f$ representing the given computational space discretization (the grid)
 * -# evaluate the given function or functor at these coordinates and store the result
 *   in the output vector \f$ v_i = f(x_i, ...)\f$ for all \c i
 * .
 *
 * For example fo a 2d grid the implementation is equivalent to
 * @code{.cpp}
 * auto abs = g.abscissas();
 * return dg::kronecker( f, abs[0], abs[1]);
 * @endcode
 * @copydoc hide_code_evaluate1d
 * @copydoc hide_code_evaluate2d
 * @copydoc hide_code_evaluate3d
 * @tparam Functor Model of Function <tt> return_type f(real_type, ...) </tt>
 * @param f The function to evaluate, see @ref functions for a host of predefined functors to evaluate
 * @param g The grid that defines the computational space on which to evaluate f
 *
 * @return The output vector \c v as a host vector
 * @note Use the elementary function \f$ f(x) = x \f$ (\c dg::cooX1d ) to generate the list of grid coordinates
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 * @sa \c dg::pullback if you want to evaluate a function in physical space
 * @note In the MPI version all processes in the grid communicator need to call
 * this function. Each process evaluates the function f only on the grid
 * coordinates that it owns i.e. the local part of the given grid
 */
template< class Functor, class Topology>
auto evaluate( Functor f, const Topology& g)
{
    return do_evaluate( f, g, std::make_index_sequence<Topology::ndim()>());
};

/// Utility function equivalent to <tt> dg::evaluate( dg::CONSTANT( value), g)</tt>
template<class Topology>
auto evaluate( enum evaluation_helper value, const Topology& g)
{
    return do_evaluate( dg::CONSTANT( value), g, std::make_index_sequence<Topology::ndim()>());
};

template<class Topology, class value_type = typename Topology::value_type, class result_type = typename Topology::value_type, typename = std::enable_if_t<Topology::ndim() == 1 > >
auto evaluate( result_type (*f)( value_type), const Topology& g)
{
    return do_evaluate( f, g, std::make_index_sequence<Topology::ndim()>());
}
template<class Topology, class value_type0 = typename Topology::value_type, class value_type1 = typename Topology::value_type, class result_type = typename Topology::value_type, typename = std::enable_if_t<Topology::ndim() == 2 > >
auto evaluate( result_type (*f)( value_type0, value_type1), const Topology& g)
{
    return do_evaluate( f, g, std::make_index_sequence<Topology::ndim()>());
}
template<class Topology, class value_type0 = typename Topology::value_type, class value_type1 = typename Topology::value_type, class value_type2 = typename Topology::value_type, class result_type = typename Topology::value_type, typename = std::enable_if_t<Topology::ndim() == 3 > >
auto evaluate( result_type (*f)( value_type0, value_type1, value_type2), const Topology& g)
{
    return do_evaluate( f, g, std::make_index_sequence<Topology::ndim()>());
}



/////////////////////////////////////INTEGRATE/////////////////

/*!@brief Indefinite integral of a function on a grid
 * \f[ F_h(x) = \int_a^x f_h(x') dx' \f]
 *
 * This function computes the indefinite integral of a given input
 * @param in Host vector discretized on g
 * @param g The grid
 * @param dir If dg::backward then the integral starts at the right boundary (i.e. goes in the reverse direction)
 * \f[ F_h(x) = \int_b^x f_h(x') dx' = \int_a^x f_h(x') dx' - \int_a^b f_h(x') dx' \f]
 * @return integral of \c in on the grid \c g
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 */
template<class real_type>
thrust::host_vector<real_type> integrate( const thrust::host_vector<real_type>& in, const RealGrid<real_type,1>& g, dg::direction dir = dg::forward)
{
    double h = g.hx();
    unsigned n = g.nx();
    thrust::host_vector<real_type> to_out(g.size(), 0.);
    thrust::host_vector<real_type> to_in(in);
    if( dir == dg::backward ) //reverse input vector
    {
        for( unsigned i=0; i<in.size(); i++)
            to_in[i] = in[ in.size()-1-i];
    }


    dg::Operator<real_type> forward = dg::DLT<real_type>::forward(n);
    dg::Operator<real_type> backward = dg::DLT<real_type>::backward(n);
    dg::Operator<real_type> ninj = create::ninj<real_type>( n );
    Operator<real_type> t = create::pipj_inv<real_type>(n);
    t *= h/2.;
    ninj = backward*t*ninj*forward;
    real_type constant = 0.;

    for( unsigned i=0; i<g.Nx(); i++)
    {
        for( unsigned k=0; k<n; k++)
        {
            for( unsigned l=0; l<n; l++)
                to_out[ i*n + k] += ninj(k,l)*to_in[ i*n + l];
            to_out[ i*n + k] += constant;
        }
        for( unsigned l=0; l<n; l++)
            constant += h*forward(0,l)*to_in[i*n+l];
    }
    thrust::host_vector<real_type> out(to_out);
    if( dir == dg::backward ) //reverse output
    {
        for( unsigned i=0; i<in.size(); i++)
            out[i] = -to_out[ in.size()-1-i]; // minus from reversing!
    }
    return out;
}


/*!@brief Indefinite integral of a function on a grid
 * \f[ F_h(x) = \int_a^x f_h(x') dx' \f]
 *
 * This function first evaluates f on the given grid and then computes
 *  and returns its indefinite integral
 * @param f The function to evaluate and then integrate
 * @param g The grid
 * @param dir If dg::backward then the integral starts at the right boundary (i.e. goes in the reverse direction)
 * \f[ F_h(x) = \int_b^x f_h(x') dx' = \int_a^x f_h(x') dx' - \int_a^b f_h(x') dx' \f]
 * @return integral of \c f on the grid \c g
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 */
template< class UnaryOp,class real_type>
thrust::host_vector<real_type> integrate( UnaryOp f, const RealGrid<real_type,1>& g, dg::direction dir = dg::forward)
{
    thrust::host_vector<real_type> vector = evaluate( f, g);
    return integrate<real_type>(vector, g, dir);
}
///@cond
template<class real_type>
thrust::host_vector<real_type> integrate( real_type (f)(real_type), const RealGrid<real_type,1>& g, dg::direction dir = dg::forward)
{
    thrust::host_vector<real_type> vector = evaluate( f, g);
    return integrate<real_type>(vector, g, dir);
};
///@endcond

///@}
}//namespace dg

