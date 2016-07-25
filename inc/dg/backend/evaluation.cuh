#ifndef _DG_EVALUATION_
#define _DG_EVALUATION_

#include <cassert> 
#include <thrust/host_vector.h>
#include "grid.h"
#include "weights.cuh"

/*! @file 
  
  Function discretization routines
  */
namespace dg
{


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
thrust::host_vector<double> evaluate( UnaryOp f, const Grid1d<double>& g)
{
    thrust::host_vector<double> abs = create::abscissas( g);
    for( unsigned i=0; i<g.size(); i++)
        abs[i] = f( abs[i]);
    return abs;
};
///@cond
thrust::host_vector<double> evaluate( double (f)(double), const Grid1d<double>& g)
{
    thrust::host_vector<double> v = evaluate<double (double)>( f, g);
    return v;
};
///@endcond


/**
 * @brief Evaluate a function on gaussian abscissas
 *
 * Evaluates f(x) on the given grid
 * @tparam BinaryOp Model of Binary Function
 * @param f The function to evaluate: f = f(x,y)
 * @param g The 2d grid on which to evaluate f
 *
 * @return  A dG Host Vector with values
 * @note Copies the binary Operator. This function is meant for small function objects, that
            may be constructed during function call.
 */
template< class BinaryOp>
thrust::host_vector<double> evaluate( BinaryOp f, const Grid2d<double>& g)
{
    unsigned n= g.n();
    //TODO: opens dlt.dat twice...!!
    Grid1d<double> gx( g.x0(), g.x1(), n, g.Nx()); 
    Grid1d<double> gy( g.y0(), g.y1(), n, g.Ny());
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
thrust::host_vector<double> evaluate( double(f)(double, double), const Grid2d<double>& g)
{
    //return evaluate<double(&)(double, double), n>( f, g );
    return evaluate<double(double, double)>( f, g);
};
///@endcond

/**
 * @brief Evaluate a function on gaussian abscissas
 *
 * Evaluates f(x,y,z) on the given grid
 * @tparam TernaryOp Model of Ternary Function
 * @param f The function to evaluate: f = f(x,y,z)
 * @param g The 3d grid on which to evaluate f
 *
 * @return  A dG Host Vector with values
 * @note Copies the ternary Operator. This function is meant for small function objects, that
            may be constructed during function call.
 */
template< class TernaryOp>
thrust::host_vector<double> evaluate( TernaryOp f, const Grid3d<double>& g)
{
    unsigned n= g.n();
    //TODO: opens dlt.dat three times...!!
    Grid1d<double> gx( g.x0(), g.x1(), n, g.Nx()); 
    Grid1d<double> gy( g.y0(), g.y1(), n, g.Ny());
    Grid1d<double> gz( g.z0(), g.z1(), 1, g.Nz());
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
thrust::host_vector<double> evaluate( double(f)(double, double, double), const Grid3d<double>& g)
{
    //return evaluate<double(&)(double, double), n>( f, g );
    return evaluate<double(double, double, double)>( f, g);
};
///@endcond
//
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
thrust::host_vector<double> evaluate( UnaryOp f, const GridX1d& g)
{
    return evaluate( f, g.grid());
};
///@cond
thrust::host_vector<double> evaluate( double (f)(double), const GridX1d& g)
{
    return evaluate( f, g.grid());
};
///@endcond


/**
 * @brief Evaluate a function on gaussian abscissas
 *
 * Evaluates f(x) on the given grid
 * @tparam BinaryOp Model of Binary Function
 * @param f The function to evaluate: f = f(x,y)
 * @param g The 2d grid on which to evaluate f
 *
 * @return  A dG Host Vector with values
 * @note Copies the binary Operator. This function is meant for small function objects, that
            may be constructed during function call.
 */
template< class BinaryOp>
thrust::host_vector<double> evaluate( BinaryOp f, const GridX2d& g)
{
    return evaluate( f, g.grid());
};
///@cond
thrust::host_vector<double> evaluate( double(f)(double, double), const GridX2d& g)
{
    return evaluate( f, g.grid());
};
///@endcond

/**
 * @brief Evaluate a function on gaussian abscissas
 *
 * Evaluates f(x,y,z) on the given grid
 * @tparam TernaryOp Model of Ternary Function
 * @param f The function to evaluate: f = f(x,y,z)
 * @param g The 3d grid on which to evaluate f
 *
 * @return  A dG Host Vector with values
 * @note Copies the ternary Operator. This function is meant for small function objects, that
            may be constructed during function call.
 */
template< class TernaryOp>
thrust::host_vector<double> evaluate( TernaryOp f, const GridX3d& g)
{
    return evaluate( f, g.grid());
};
///@cond
thrust::host_vector<double> evaluate( double(f)(double, double, double), const GridX3d& g)
{
    return evaluate( f, g.grid());
};
///@endcond

///@}
}//namespace dg

#endif //_DG_EVALUATION
