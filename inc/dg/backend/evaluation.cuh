#pragma once

#include <cassert> 
#include <thrust/host_vector.h>
#include "grid.h"
#include "weights.cuh"

/*! @file 
  @brief Function discretization routines
  */
namespace dg
{

//maybe we should consider using boost::function as argument type as a generic function pointer typ

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
 */
template< class UnaryOp>
thrust::host_vector<double> evaluate( UnaryOp f, const Grid1d& g)
{
    thrust::host_vector<double> abs = create::abscissas( g);
    for( unsigned i=0; i<g.size(); i++)
        abs[i] = f( abs[i]);
    return abs;
};
///@cond
thrust::host_vector<double> evaluate( double (f)(double), const Grid1d& g)
{
    thrust::host_vector<double> v = evaluate<double (double)>( *f, g);
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
 * @note if you don't like to copy f then just pass a (const) reference then the type should adapt
 */
template< class BinaryOp>
thrust::host_vector<double> evaluate( const BinaryOp& f, const aTopology2d& g)
{
    unsigned n= g.n();
    Grid1d gx(g.x0(), g.x1(), g.n(), g.Nx());
    Grid1d gy(g.y0(), g.y1(), g.n(), g.Ny());
    thrust::host_vector<double> absx = create::abscissas( gx);
    thrust::host_vector<double> absy = create::abscissas( gy);

    //choose layout in the comments
    thrust::host_vector<double> v( g.size());
    for( unsigned i=0; i<gy.N(); i++)
        for( unsigned k=0; k<n; k++)
            for( unsigned j=0; j<gx.N(); j++)
                for( unsigned l=0; l<n; l++)
                    //v[ i*g.Nx()*n*n + j*n*n + k*n + l] = f( absx[j*n+l], absy[i*n+k]);
                    v[ ((i*n+k)*g.Nx() + j)*n + l] = f( absx[j*n+l], absy[i*n+k]);
    return v;
};
///@cond
thrust::host_vector<double> evaluate( double(f)(double, double), const aTopology2d& g)
{
    //return evaluate<double(&)(double, double), n>( f, g );
    return evaluate<double(double, double)>( *f, g);
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
 * @note if you don't like to copy f then just pass a (const) reference then the type should adapt
 */
template< class TernaryOp>
thrust::host_vector<double> evaluate( const TernaryOp& f, const aTopology3d& g)
{
    unsigned n= g.n();
    Grid1d gx(g.x0(), g.x1(), g.n(), g.Nx());
    Grid1d gy(g.y0(), g.y1(), g.n(), g.Ny());
    Grid1d gz(g.z0(), g.z1(), 1, g.Nz());
    thrust::host_vector<double> absx = create::abscissas( gx);
    thrust::host_vector<double> absy = create::abscissas( gy);
    thrust::host_vector<double> absz = create::abscissas( gz);

    thrust::host_vector<double> v( g.size());
    for( unsigned s=0; s<gz.N(); s++)
        for( unsigned i=0; i<gy.N(); i++)
            for( unsigned k=0; k<n; k++)
                for( unsigned j=0; j<gx.N(); j++)
                    for( unsigned l=0; l<n; l++)
                        //v[ s*g.Nx()*g.Ny()*n*n + i*g.Nx()*n*n + j*n*n + k*n + l] = f( absx[j*n+l], absy[i*n+k], absz[s]);
                        v[ (((s*gy.N()+i)*n+k)*g.Nx() + j)*n + l] = f( absx[j*n+l], absy[i*n+k], absz[s]);
    return v;
};
///@cond
thrust::host_vector<double> evaluate( double(f)(double, double, double), const aTopology3d& g)
{
    //return evaluate<double(&)(double, double), n>( f, g );
    return evaluate<double(double, double, double)>( *f, g);
};
///@endcond

///@}
}//namespace dg

