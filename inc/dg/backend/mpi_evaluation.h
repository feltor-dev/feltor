#pragma once

#include "mpi_grid.h"
#include "mpi_vector.h"
#include "evaluation.cuh"

/*! @file 
  
  Function discretization routines for mpi vectors
  */
namespace dg
{


///@addtogroup evaluation
///@{



/**
 * @brief Evaluate a function on gaussian abscissas
 *
 * Evaluates f(x) on the intervall (a,b)
 * @tparam Function Model of Unary Function
 * @param f The function to evaluate
 * @param g The grid on which to evaluate f
 *
 * @return  A MPI Vector with values
 */
    /*
template< class Function>
MPI_Vector evaluate( Function f, const MPIGrid1d<double>& g)
{
    MPI_Vector v;
    v.Nx = g.Nx();
    thrust::host_vector<double> abs = create::abscissas( g);
    for( unsigned i=0; i<g.size(); i++)
        abs[i] = f( abs[i]);
    return abs;
};
///@cond
MPI_Vector evaluate( double (f)(double), const Grid1d<double>& g)
{
    thrust::host_vector<double> v = evaluate<double (double)>( f, g);
    return v;
};
///@endcond
*/


/**
 * @brief Evaluate a function on gaussian abscissas
 *
 * Evaluates f(x) on the given grid
 * @tparam Function Model of Binary Function
 * @param f The function to evaluate: f = f(x,y)
 * @param g The 2d grid on which to evaluate f
 *
 * @return  A MPI Host Vector with values
 * @note Copies the binary Operator. This function is meant for small function objects, that
            may be constructed during function call.
 */
template< class BinaryOp>
MPI_Vector evaluate( BinaryOp f, const MPI_Grid2d& g)
{
    MPI_Vector v( g.n()*g.n(), g.Nx(), g.Ny());
    v.data() = evaluate(f,g.local());
    return v;
};
///@cond
MPI_Vector evaluate( double(f)(double, double), const MPI_Grid2d& g)
{
    return evaluate<double(double, double)>( f, g);
};
///@endcond

/**
 * @brief Evaluate a function on gaussian abscissas
 *
 * Evaluates f(x,y,z) on the given grid
 * @tparam Function Model of Ternary Function
 * @param f The function to evaluate: f = f(x,y,z)
 * @param g The 3d grid on which to evaluate f
 *
 * @return  A MPI Vector with values
 * @note Copies the ternary Operator. This function is meant for small function objects, that
            may be constructed during function call.
 */
/*
template< class TernaryOp>
MPI_Vector evaluate( TernaryOp f, const MPIGrid3d<double>& g)
{
};
///@cond
MPI_Vector evaluate( double(f)(double, double, double), const MPIGrid3d<double>& g)
{
    return evaluate<double(double, double, double)>( f, g);
};
///@endcond
*/


///@}
}//namespace dg

