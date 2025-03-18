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
auto do_evaluate( Functor&& f, const Topology& g, std::index_sequence<I...>)
{
    return dg::kronecker( std::forward<Functor>(f), g.abscissas(I)...);
}

///@endcond
///@addtogroup evaluation
///@{

/**
 * @brief %Evaluate a function on grid coordinates
 *
 * %Evaluate is equivalent to the following:
 *
 * -# from the given Nd dimensional grid generate Nd one-dimensional lists of
 *  grid coordinates <tt>x_i = g.abscissas( i)</tt> representing the given
 *  computational space discretization in each dimension
 * -# evaluate the given function or functor at these coordinates and store the
 *  result in the output vector <tt> v = dg::kronecker( f, x_0, x_1, ...)</tt>
 *  The dimension number \c i is thus mapped to the argument number of the
 *  function \c f. The **0 dimension is the contiguous dimension** in the
 *  return vector \c v e.g. in 2D the first element of the resulting vector
 *  lies in the grid corner \f$ (x_0,y_0)\f$, the second is \f$(x_1, y_0)\f$
 *  and so on.
 * .
 *
 * For example for a 2d grid the implementation is equivalent to
 * @code{.cpp}
 * return dg::kronecker( f, g.abscissas(0), g.abscissas(1));
 * @endcode
 *
 * See here an application example
 * @snippet{trimleft} evaluation_t.cpp evaluate2d
 * @tparam Topology A fixed sized grid type with member functions <tt> static
 * constexpr size_t Topology::ndim()</tt> giving the number of dimensions and
 * <tt> vector_type Topology::abscissas( unsigned dim)</tt> giving the
 * abscissas in dimension \c dim
 * @tparam Functor Callable as <tt> return_type f(real_type, ...)</tt>.
 * \c Functor needs to be callable with \c Topology::ndim arguments.
 * @param f The function to evaluate, see @ref functions for a host of
 * predefined functors to evaluate
 * @param g The grid that defines the computational space on which to evaluate
 * \c f
 *
 * @return The output vector \c v as a host vector. Its value type is
 * determined by the return type of \c Functor
 * @note Use the elementary function \f$ f(x) = x \f$ (\c dg::cooX1d ) to
 * generate the list of grid coordinates
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 * @sa \c dg::pullback if you want to evaluate a function in physical space
 * @sa \c dg::kronecker
 * @note In the MPI version all processes in the grid communicator need to call
 * this function. Each process evaluates the function f only on the grid
 * coordinates that it owns i.e. the local part of the given grid
 */
template< class Functor, class Topology>
auto evaluate( Functor&& f, const Topology& g)
{
    // The evaluate function is the reason why our Topology needs to have fixed
    // sized dimensions instead of dynamically sized dimensions
    // even though if we really wanted we could maybe ask ndim = g.ndim()
    // and then do use switch and manually implement until ndim < 10 say
    // for now we keep fixed sized grids ...
    //
    // If we ever change the order of the fastest dimension we need to rethink
    // NetCDF hyperslab construction
    return do_evaluate( std::forward<Functor>(f), g, std::make_index_sequence<Topology::ndim()>());
};

///@cond
//These overloads help the compiler in a situation where a free function has
//several overloads of different dimensions e.g.
//double zero( double);
//double zero( double,double);
//In such a case dg::evaluate( zero, grid); cannot determine the Functor type...
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
///@endcond



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


    dg::SquareMatrix<real_type> forward = dg::DLT<real_type>::forward(n);
    dg::SquareMatrix<real_type> backward = dg::DLT<real_type>::backward(n);
    dg::SquareMatrix<real_type> ninj = create::ninj<real_type>( n );
    SquareMatrix<real_type> t = create::pipj_inv<real_type>(n);
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


/*!@brief Untility shortcut
 *
 * for
 * @code{.cpp}
 *   thrust::host_vector<real_type> vector = evaluate( f, g);
 *   return integrate<real_type>(vector, g, dir);
 *  @endcode
 *
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

