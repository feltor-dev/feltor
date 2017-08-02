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
 * Evaluates f(x) on the given grid
 * @tparam Function Model of Binary Function
 * @param f The function to evaluate: f = f(x,y)
 * @param g The 2d grid on which to evaluate f
 *
 * @return  A MPI Vector with values
 * @note Copies the binary Operator. This function is meant for small function objects, that
            may be constructed during function call.
 */
template< class BinaryOp>
MPI_Vector<thrust::host_vector<double> > evaluate( BinaryOp f, const aMPITopology2d& g)
{
    thrust::host_vector<double> w = evaluate( f, g.local());
    MPI_Vector<thrust::host_vector<double> > v( w, g.communicator());
    //v.data() = evaluate(f,g.local());
    return v;
};
///@cond
MPI_Vector<thrust::host_vector<double> > evaluate( double(f)(double, double), const aMPITopology2d& g)
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
template< class TernaryOp>
MPI_Vector<thrust::host_vector<double> > evaluate( TernaryOp f, const aMPITopology3d& g)
{
    thrust::host_vector<double> w = evaluate( f, g.local());
    MPI_Vector<thrust::host_vector<double> > v( w, g.communicator());
    //v.data() = evaluate(f, g.local());
    return v;
};
///@cond
MPI_Vector<thrust::host_vector<double> > evaluate( double(f)(double, double, double), const aMPITopology3d& g)
{
    return evaluate<double(double, double, double)>( f, g);
};
///@endcond


///@}
}//namespace dg

