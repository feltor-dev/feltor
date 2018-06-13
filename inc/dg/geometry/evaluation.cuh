#pragma once

#include <cassert>
#include <cmath>
#include <thrust/host_vector.h>
#include "dg/backend/config.h"
#include "grid.h"

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
thrust::host_vector<real_type> abscissas( const BasicGrid1d<real_type>& g)
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
 * @brief Evaluate a function on gaussian abscissas
 *
 * Evaluates f(x) on the intervall (a,b)
 * @tparam UnaryOp Model of Unary Function
 * @param f The function to evaluate
 * @param g The grid on which to evaluate f
 *
 * @return  A DG Host Vector with values
 * @sa <a href="./dg_introduction.pdf" target="_blank">Introduction to dg methods</a>
 * @copydoc hide_code_evaluate1d
 */
template< class UnaryOp,class real_type>
thrust::host_vector<real_type> evaluate( UnaryOp f, const BasicGrid1d<real_type>& g)
{
    thrust::host_vector<real_type> abs = create::abscissas( g);
    for( unsigned i=0; i<g.size(); i++)
        abs[i] = f( abs[i]);
    return abs;
};
///@cond
template<class real_type>
thrust::host_vector<real_type> evaluate( real_type (f)(real_type), const BasicGrid1d<real_type>& g)
{
    thrust::host_vector<real_type> v = evaluate<real_type (real_type)>( *f, g);
    return v;
};
///@endcond


/**
 * @brief Evaluate a function on gaussian abscissas
 *
 * Evaluates f(x) on the given grid
 * @copydoc hide_binary
 * @param f The function to evaluate: f = f(x,y)
 * @param g The 2d grid on which to evaluate f
 *
 * @return  A dG Host Vector with values
 * @sa <a href="./dg_introduction.pdf" target="_blank">Introduction to dg methods</a>
 * @copydoc hide_code_evaluate2d
 */
template< class BinaryOp, class real_type>
thrust::host_vector<real_type> evaluate( const BinaryOp& f, const aBasicTopology2d<real_type>& g)
{
    unsigned n= g.n();
    BasicGrid1d<real_type> gx(g.x0(), g.x1(), g.n(), g.Nx());
    BasicGrid1d<real_type> gy(g.y0(), g.y1(), g.n(), g.Ny());
    thrust::host_vector<real_type> absx = create::abscissas( gx);
    thrust::host_vector<real_type> absy = create::abscissas( gy);

    thrust::host_vector<real_type> v( g.size());
    for( unsigned i=0; i<gy.N(); i++)
        for( unsigned k=0; k<n; k++)
            for( unsigned j=0; j<gx.N(); j++)
                for( unsigned r=0; r<n; r++)
                    v[ ((i*n+k)*g.Nx() + j)*n + r] = f( absx[j*n+r], absy[i*n+k]);
    return v;
};
///@cond
template<class real_type>
thrust::host_vector<real_type> evaluate( real_type(f)(real_type, real_type), const aBasicTopology2d<real_type>& g)
{
    return evaluate<real_type(real_type, real_type)>( *f, g);
};
///@endcond

/**
 * @brief Evaluate a function on gaussian abscissas
 *
 * Evaluates f(x,y,z) on the given grid
 * @copydoc hide_ternary
 * @param f The function to evaluate: f = f(x,y,z)
 * @param g The 3d grid on which to evaluate f
 *
 * @return  A dG Host Vector with values
 * @sa <a href="./dg_introduction.pdf" target="_blank">Introduction to dg methods</a>
 * @copydoc hide_code_evaluate3d
 */
template< class TernaryOp,class real_type>
thrust::host_vector<real_type> evaluate( const TernaryOp& f, const aBasicTopology3d<real_type>& g)
{
    unsigned n= g.n();
    BasicGrid1d<real_type> gx(g.x0(), g.x1(), g.n(), g.Nx());
    BasicGrid1d<real_type> gy(g.y0(), g.y1(), g.n(), g.Ny());
    BasicGrid1d<real_type> gz(g.z0(), g.z1(), 1, g.Nz());
    thrust::host_vector<real_type> absx = create::abscissas( gx);
    thrust::host_vector<real_type> absy = create::abscissas( gy);
    thrust::host_vector<real_type> absz = create::abscissas( gz);

    thrust::host_vector<real_type> v( g.size());
    for( unsigned s=0; s<gz.N(); s++)
        for( unsigned i=0; i<gy.N(); i++)
            for( unsigned k=0; k<n; k++)
                for( unsigned j=0; j<gx.N(); j++)
                    for( unsigned l=0; l<n; l++)
                        v[ (((s*gy.N()+i)*n+k)*g.Nx() + j)*n + l] = f( absx[j*n+l], absy[i*n+k], absz[s]);
    return v;
};
///@cond
template<class real_type>
thrust::host_vector<real_type> evaluate( real_type(f)(real_type, real_type, real_type), const aBasicTopology3d<real_type>& g)
{
    return evaluate<real_type(real_type, real_type, real_type)>( *f, g);
};
///@endcond

///@}
}//namespace dg

