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
namespace create
{
/**
* @brief create host_vector containing 1d X-space abscissas
*
* same as evaluation of f(x) = x on the grid
* @param g The grid
*
* @return Host Vector
*/
template<class real_type>
thrust::host_vector<real_type> abscissas( const RealGrid1d<real_type>& g)
{
    thrust::host_vector<real_type> abs(g.size());
    for( unsigned i=0; i<g.N(); i++)
        for( unsigned j=0; j<g.n(); j++)
        {
            real_type xmiddle = DG_FMA( g.h(), (real_type)(i), g.x0());
            real_type h2 = g.h()/2.;
            real_type absj = 1.+g.dlt().abscissas()[j];
            abs[i*g.n()+j] = DG_FMA( h2, absj, xmiddle);
        }
    return abs;
}
}//
///@endcond

///@addtogroup evaluation
///@{


/**
 * @brief Evaluate a 1d function on grid coordinates
 *
 * Evaluate is equivalent to the following:
 *
 * -# generate a list of grid coordinates \f$ x_i\f$ representing the given computational space discretization (the grid)
 * -# evaluate the given function or functor at these coordinates and store the result
 *   in the output vector \f$ v_i = f(x_i)\f$ for all \c i
 * .
 * @copydoc hide_code_evaluate1d
 * @tparam UnaryOp Model of Unary Function <tt> real_type f(real_type) </tt>
 * @param f The function to evaluate, see @ref functions for a host of predefined functors to evaluate
 * @param g The grid that defines the computational space on which to evaluate f
 *
 * @return The output vector \c v as a host vector
 * @note Use the elementary function \f$ f(x) = x \f$ (\c dg::cooX1d ) to generate the list of grid coordinates
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 * @sa \c dg::pullback if you want to evaluate a function in physical space
 */
template< class UnaryOp,class real_type>
thrust::host_vector<real_type> evaluate( UnaryOp f, const RealGrid1d<real_type>& g)
{
    thrust::host_vector<real_type> abs = create::abscissas( g);
    for( unsigned i=0; i<g.size(); i++)
        abs[i] = f( abs[i]);
    return abs;
};
///@cond
template<class real_type>
thrust::host_vector<real_type> evaluate( real_type (f)(real_type), const RealGrid1d<real_type>& g)
{
    thrust::host_vector<real_type> v = evaluate<real_type (real_type)>( *f, g);
    return v;
};
///@endcond


/**
 * @brief Evaluate a 2d function on grid coordinates
 *
 * Evaluate is equivalent to the following:
 *
 * -# generate the list of grid coordinates \f$ x_i\f$, \f$ y_i\f$ representing the given computational space discretization (the grid)
 * -# evaluate the given function or functor at these coordinates and store the result
 *   in the output vector \f$ v_i = f(x_i, y_i)\f$ for all \f$ i \f$
 *.
 * @copydoc hide_code_evaluate2d
 * @copydoc hide_binary
 * @param f The function to evaluate: \f$ f = f(x,y)\f$, see @ref functions for a host of predefined functors to evaluate
 * @param g The 2d grid on which to evaluate \c f
 *
 * @return The output vector \c v as a host vector
 * @note Use the elementary function \f$ f(x,y) = x \f$ (\c dg::cooX2d) to generate the list of grid coordinates in \c x direction (or analogous in \c y, \c dg::cooY2d)
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 * @sa \c dg::pullback if you want to evaluate a function in physical space
 */
template< class BinaryOp, class real_type>
thrust::host_vector<real_type> evaluate( const BinaryOp& f, const aRealTopology2d<real_type>& g)
{
    thrust::host_vector<real_type> absx = create::abscissas( g.gx());
    thrust::host_vector<real_type> absy = create::abscissas( g.gy());

    thrust::host_vector<real_type> v( g.size());
    for( unsigned i=0; i<g.Ny(); i++)
    for( unsigned k=0; k<g.ny(); k++)
    for( unsigned j=0; j<g.Nx(); j++)
    for( unsigned r=0; r<g.nx(); r++)
        v[ ((i*g.ny()+k)*g.Nx() + j)*g.nx() + r] =
                f( absx[j*g.nx()+r], absy[i*g.ny()+k]);
    return v;
};
///@cond
template<class real_type>
thrust::host_vector<real_type> evaluate( real_type(f)(real_type, real_type), const aRealTopology2d<real_type>& g)
{
    return evaluate<real_type(real_type, real_type)>( *f, g);
};
///@endcond

/**
 * @brief Evaluate a 3d function on grid coordinates
 *
 * Evaluate is equivalent to the following:
 *
 * -# generate the list of grid coordinates \f$ x_i\f$, \f$ y_i\f$, \f$ z_i \f$ representing the given computational space discretization (the grid)
 * -# evaluate the given function or functor at these coordinates and store the result
 *   in the output vector \f$ v_i = f(x_i, y_i, z_i)\f$ for all \f$ i\f$
 *.
 * @copydoc hide_code_evaluate3d
 * @copydoc hide_ternary
 * @param f The function to evaluate: \f$ f = f(x,y,z) \f$, see @ref functions for a host of predefined functors to evaluate
 * @param g The 3d grid on which to evaluate \c f
 *
 * @return The output vector \c v as a host vector
 * @note Use the elementary function \f$ f(x,y,z) = x \f$ (\c dg::cooX3d) to generate the list of grid coordinates in \c x direction (or analogous in \c y, \c dg::cooY3d or \c z, \c dg::cooZ3d)
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 * @sa \c dg::pullback if you want to evaluate a function in physical space
 */
template< class TernaryOp,class real_type>
thrust::host_vector<real_type> evaluate( const TernaryOp& f, const aRealTopology3d<real_type>& g)
{
    thrust::host_vector<real_type> absx = create::abscissas( g.gx());
    thrust::host_vector<real_type> absy = create::abscissas( g.gy());
    thrust::host_vector<real_type> absz = create::abscissas( g.gz());

    thrust::host_vector<real_type> v( g.size());
    for( unsigned s=0; s<g.Nz(); s++)
    for( unsigned ss=0; ss<g.nz(); ss++)
    for( unsigned i=0; i<g.Ny(); i++)
    for( unsigned ii=0; ii<g.ny(); ii++)
    for( unsigned k=0; k<g.Nx(); k++)
    for( unsigned kk=0; kk<g.nx(); kk++)
        v[ ((((s*g.nz()+ss)*g.Ny()+i)*g.ny()+ii)*g.Nx() + k)*g.nx() + kk] =
            f( absx[k*g.nx()+kk], absy[i*g.ny()+ii], absz[s*g.nz()+ss]);
    return v;
};
///@cond
template<class real_type>
thrust::host_vector<real_type> evaluate( real_type(f)(real_type, real_type, real_type), const aRealTopology3d<real_type>& g)
{
    return evaluate<real_type(real_type, real_type, real_type)>( *f, g);
};
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
thrust::host_vector<real_type> integrate( const thrust::host_vector<real_type>& in, const RealGrid1d<real_type>& g, dg::direction dir = dg::forward)
{
    double h = g.h();
    unsigned n = g.n();
    thrust::host_vector<real_type> to_out(g.size(), 0.);
    thrust::host_vector<real_type> to_in(in);
    if( dir == dg::backward ) //reverse input vector
    {
        for( unsigned i=0; i<in.size(); i++)
            to_in[i] = in[ in.size()-1-i];
    }


    dg::Operator<real_type> forward = g.dlt().forward();
    dg::Operator<real_type> backward = g.dlt().backward();
    dg::Operator<real_type> ninj = create::ninj<real_type>( n );
    Operator<real_type> t = create::pipj_inv<real_type>(n);
    t *= h/2.;
    ninj = backward*t*ninj*forward;
    real_type constant = 0.;

    for( unsigned i=0; i<g.N(); i++)
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
thrust::host_vector<real_type> integrate( UnaryOp f, const RealGrid1d<real_type>& g, dg::direction dir = dg::forward)
{
    thrust::host_vector<real_type> vector = evaluate( f, g);
    return integrate<real_type>(vector, g, dir);
}
///@cond
template<class real_type>
thrust::host_vector<real_type> integrate( real_type (f)(real_type), const RealGrid1d<real_type>& g, dg::direction dir = dg::forward)
{
    thrust::host_vector<real_type> vector = evaluate( f, g);
    return integrate<real_type>(vector, g, dir);
};
///@endcond

///@}
}//namespace dg

