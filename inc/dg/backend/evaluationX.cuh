#pragma once

#include "gridX.h"
#include "evaluation.cuh"


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
thrust::host_vector<double> evaluate( BinaryOp f, const aTopologyX2d& g)
{
    return evaluate( f, g.grid());
};
///@cond
thrust::host_vector<double> evaluate( double(f)(double, double), const aTopologyX2d& g)
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
thrust::host_vector<double> evaluate( TernaryOp f, const aTopologyX3d& g)
{
    return evaluate( f, g.grid());
};
///@cond
thrust::host_vector<double> evaluate( double(f)(double, double, double), const aTopologyX3d& g)
{
    return evaluate( f, g.grid());
};
///@endcond

///@}
}//namespace dg

