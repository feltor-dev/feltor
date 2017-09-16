#pragma once

#include "mpi_grid.h"
#include "mpi_vector.h"
#include "evaluation.cuh"

/*! @file 
  @brief Function discretization routines for mpi vectors
  */
namespace dg
{


///@addtogroup evaluation
///@{

/**
 * @brief Evaluate a function on gaussian abscissas
 *
 * Evaluates f(x) on the given grid
 * @copydoc hide_binary
 * @param f The function to evaluate: f = f(x,y)
 * @param g The 2d grid on which to evaluate f
 *
 * @return  A MPI Vector with values
 * @note Copies the binary Operator. This function is meant for small function objects, that
            may be constructed during function call.
 */
template< class BinaryOp>
MPI_Vector<thrust::host_vector<double> > evaluate( const BinaryOp& f, const aMPITopology2d& g)
{
    thrust::host_vector<double> w = evaluate( f, g.local());
    MPI_Vector<thrust::host_vector<double> > v( w, g.communicator());
    //v.data() = evaluate(f,g.local());
    return v;
};
///@cond
MPI_Vector<thrust::host_vector<double> > evaluate( double(f)(double, double), const aMPITopology2d& g)
{
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
 * @return  A MPI Vector with values
 * @note Copies the ternary Operator. This function is meant for small function objects, that
            may be constructed during function call.
 */
template< class TernaryOp>
MPI_Vector<thrust::host_vector<double> > evaluate( const TernaryOp& f, const aMPITopology3d& g)
{
    thrust::host_vector<double> w = evaluate( f, g.local());
    MPI_Vector<thrust::host_vector<double> > v( w, g.communicator());
    //v.data() = evaluate(f, g.local());
    return v;
};
///@cond
MPI_Vector<thrust::host_vector<double> > evaluate( double(f)(double, double, double), const aMPITopology3d& g)
{
    return evaluate<double(double, double, double)>( *f, g);
};
///@endcond
//
///@}

/**
 * @brief Take the relevant local part of a global vector
 *
 * @param global a vector the size of the global grid 
 * @param g the assumed topology
 * @return an MPI_Vector that is the distributed version of the global vector
 * @ingroup scatter
 */
MPI_Vector<thrust::host_vector<double> > global2local( const thrust::host_vector<double>& global, const aMPITopology3d& g)
{
    assert( global.size() == g.global().size());
    thrust::host_vector<double> temp(g.size());
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( g.communicator(), 3, dims, periods, coords);
    for( unsigned s=0; s<g.Nz(); s++)
        //for( unsigned py=0; py<dims[1]; py++)
            for( unsigned i=0; i<g.n()*g.Ny(); i++)
                //for( unsigned px=0; px<dims[0]; px++)
                    for( unsigned j=0; j<g.n()*g.Nx(); j++)
                    {
                        unsigned idx1 = (s*g.n()*g.Ny()+i)*g.n()*g.Nx() + j;
                        unsigned idx2 = (((s*dims[1]+coords[1])*g.n()*g.Ny()+i)*dims[0] + coords[0])*g.n()*g.Nx() + j;
                        temp[idx1] = global[idx2];
                    }
    return MPI_Vector<thrust::host_vector<double> >(temp, g.communicator()); 
}
/**
 * @brief Take the relevant local part of a global vector
 *
 * @param global a vector the size of the global grid 
 * @param g the assumed topology
 * @return an MPI_Vector that is the distributed version of the global vector
 * @ingroup scatter
 */
MPI_Vector<thrust::host_vector<double> > global2local( const thrust::host_vector<double>& global, const aMPITopology2d& g)
{
    assert( global.size() == g.global().size());
    thrust::host_vector<double> temp(g.size());
    int dims[2], periods[2], coords[2];
    MPI_Cart_get( g.communicator(), 2, dims, periods, coords);
    //for( unsigned py=0; py<dims[1]; py++)
        for( unsigned i=0; i<g.n()*g.Ny(); i++)
            //for( unsigned px=0; px<dims[0]; px++)
                for( unsigned j=0; j<g.n()*g.Nx(); j++)
                {
                    unsigned idx1 = i*g.n()*g.Nx() + j;
                    unsigned idx2 = ((coords[1]*g.n()*g.Ny()+i)*dims[0] + coords[0])*g.n()*g.Nx() + j;
                    temp[idx1] = global[idx2];
                }
    return MPI_Vector<thrust::host_vector<double> >(temp, g.communicator()); 
}

}//namespace dg

