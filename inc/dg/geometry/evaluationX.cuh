#pragma once

#include "gridX.h"
#include "evaluation.cuh"


/*! @file
  @brief Function discretization routines on X-point topology
  */
namespace dg
{

///@cond
namespace create{
template<class real_type>
thrust::host_vector<real_type> abscissas( const RealGridX1d<real_type>& g)
{
    return abscissas(g.grid());
}
}//namespace create
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
 */
template< class UnaryOp,class real_type>
thrust::host_vector<real_type> evaluate( UnaryOp f, const RealGridX1d<real_type>& g)
{
    return evaluate( f, g.grid());
};
///@cond
template<class real_type>
thrust::host_vector<real_type> evaluate( real_type (f)(real_type), const RealGridX1d<real_type>& g)
{
    return evaluate( *f, g.grid());
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
 * @note Copies the binary Operator. This function is meant for small function objects, that
            may be constructed during function call.
 */
template< class BinaryOp, class real_type>
thrust::host_vector<real_type> evaluate( const BinaryOp& f, const aRealTopologyX2d<real_type>& g)
{
    return evaluate( f, g.grid());
};
///@cond
template<class real_type>
thrust::host_vector<real_type> evaluate( real_type(f)(real_type, real_type), const aRealTopologyX2d<real_type>& g)
{
    return evaluate( *f, g.grid());
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
 * @note Copies the ternary Operator. This function is meant for small function objects, that
            may be constructed during function call.
 */
template< class TernaryOp, class real_type>
thrust::host_vector<real_type> evaluate( const TernaryOp& f, const aRealTopologyX3d<real_type>& g)
{
    return evaluate( f, g.grid());
};
///@cond
template<class real_type>
thrust::host_vector<real_type> evaluate( real_type(f)(real_type, real_type, real_type), const aRealTopologyX3d<real_type>& g)
{
    return evaluate( *f, g.grid());
};
///@endcond

///@}
}//namespace dg

