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
 * @sa <a href="./dg_introduction.pdf" target="_blank">Introduction to dg methods</a>
 * @copydoc hide_code_mpi_evaluate2d
 */
template< class BinaryOp>
MPI_Vector<thrust::host_vector<double> > evaluate( const BinaryOp& f, const aMPITopology2d& g)
{
    thrust::host_vector<double> w = evaluate( f, g.local());
    MPI_Vector<thrust::host_vector<double> > v( w, g.communicator());
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
 * @sa <a href="./dg_introduction.pdf" target="_blank">Introduction to dg methods</a>
 * @copydoc hide_code_mpi_evaluate3d
 */
template< class TernaryOp>
MPI_Vector<thrust::host_vector<double> > evaluate( const TernaryOp& f, const aMPITopology3d& g)
{
    thrust::host_vector<double> w = evaluate( f, g.local());
    MPI_Vector<thrust::host_vector<double> > v( w, g.communicator());
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
    Grid3d l = g.local();
    thrust::host_vector<double> temp(l.size());
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( g.communicator(), 3, dims, periods, coords);
    for( unsigned s=0; s<l.Nz(); s++)
        //for( unsigned py=0; py<dims[1]; py++)
            for( unsigned i=0; i<l.n()*l.Ny(); i++)
                //for( unsigned px=0; px<dims[0]; px++)
                    for( unsigned j=0; j<l.n()*l.Nx(); j++)
                    {
                        unsigned idx1 = (s*l.n()*l.Ny()+i)*l.n()*l.Nx() + j;
                        unsigned idx2 = (((s*dims[1]+coords[1])*l.n()*l.Ny()+i)*dims[0] + coords[0])*l.n()*l.Nx() + j;
                        temp[idx1] = global[idx2];
                    }
    return MPI_Vector<thrust::host_vector<double> >(temp, g.communicator());
}
/**
 * @copydoc global2local
 * @ingroup scatter
 */
MPI_Vector<thrust::host_vector<double> > global2local( const thrust::host_vector<double>& global, const aMPITopology2d& g)
{
    assert( global.size() == g.global().size());
    Grid2d l = g.local();
    thrust::host_vector<double> temp(l.size());
    int dims[2], periods[2], coords[2];
    MPI_Cart_get( g.communicator(), 2, dims, periods, coords);
    //for( unsigned py=0; py<dims[1]; py++)
        for( unsigned i=0; i<l.n()*l.Ny(); i++)
            //for( unsigned px=0; px<dims[0]; px++)
                for( unsigned j=0; j<l.n()*l.Nx(); j++)
                {
                    unsigned idx1 = i*l.n()*l.Nx() + j;
                    unsigned idx2 = ((coords[1]*l.n()*l.Ny()+i)*dims[0] + coords[0])*l.n()*l.Nx() + j;
                    temp[idx1] = global[idx2];
                }
    return MPI_Vector<thrust::host_vector<double> >(temp, g.communicator());
}

}//namespace dg

